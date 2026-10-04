# Local Diffusion

Run open text-to-image checkpoints locally - **FLUX.1 Krea [dev]**, **Krea 2
Turbo**, FLUX.2 [klein], Z-Image, ERNIE-Image, Qwen-Image, Animagine, Kohaku, Illustrious, WAI,
Z-Anime, Anima Turbo, NetaYume Lumina and NewBie - up to
4 megapixels on a single consumer GPU. FastAPI + diffusers backend, vanilla-JS
frontend, SQLite gallery. FLUX is the default because its repo is click-through
gated; Turbo needs approval first.

![stack](https://img.shields.io/badge/torch-2.13%2Bcu130-informational)
![stack](https://img.shields.io/badge/diffusers-0.40.0-informational)

Choosing an anime checkpoint? See the [anime model guide](docs/anime-model-guide.md)
for a quick chooser, prompting differences, starting settings and measured speeds.

## Requirements

- NVIDIA GPU, **16 GB VRAM minimum** (float8 path). 24 GB+ can run `LOCAL_DIFFUSION_QUANT=none`.
- Free disk varies by checkpoint: FLUX.1 Krea [dev] is 32 GiB (transformer
  23.8 GB, T5 9.5 GB), Krea 2 Turbo 34 GiB, and Animagine's Diffusers
  components total about 6.5 GiB. Illustrious and WAI each need about 14 GB for
  their source checkpoint and converted component tree. Z-Anime plus its standard Qwen3
  encoder is about 20.7 GB (19.3 GiB). The WSL Nunchaku Lite re-uploads are far
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
Turbo, Qwen-Image, ERNIE-Image Turbo, Animagine XL 4.0, Kohaku XL Zeta, Illustrious XL v2.0,
Z-Anime Base and Anima Turbo are open and need no login (Apache 2.0 except
Animagine's Open RAIL++-M, Kohaku's Fair AI Public License 1.0-SD,
Illustrious's CreativeML OpenRAIL-M and Anima's
[CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima#license);
klein is the cheapest thing
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

The loader resolves the pipeline class and quantizable components from the
repo's `model_index.json`. Illustrious, WAI, Anima Turbo, NetaYume and NewBie first
convert pinned weights to a local component tree, then use the same loading
and offload path. WAI's source is a hash-checked community mirror of the
author's checkpoint; the other converted entries use official repositories.
If a conversion is interrupted, the next load of that checkpoint removes its
partial tree.

## Run

```powershell
uv sync                            # first time only
uv run python -m backend.main
```

Started bare on a terminal, it asks which checkpoint, quantization and offload
to use - numbered lists, Enter keeps the marked default, and a repo id that is
not listed can be pasted in. It then prints the `--model/--quant/--offload`
line that would start the same thing without asking. Any argument at all
skips the menu, and whatever it does not name comes from the `LOCAL_DIFFUSION_*`
variables below:

```powershell
uv run python -m backend.main --defaults               # the environment, no questions
uv run python -m backend.main --model krea/Krea-2-Turbo --quant int8
```

For **WAI Illustrious v17**, choose it in the menu or run:

```powershell
uv run python -m backend.main --model LyliaEngine/waiIllustriousSDXL_v170 --quant fp8 --offload model
```

The app starts at **28 steps and CFG 6**, within the author's 15-30 steps and
CFG 5-7 guidance, with Euler Ancestral and Mu disabled. Use tags or a short
description; **Tags** can append the author's quality recipe and fill the
recommended negative prompt. **Examples** contains that quality recipe alone,
so add your subject before generating. The author recommends original sizes
above 1024x1024 and uses 1024x1344 before a separate 1.5x hires pass. Generate
renders directly at the requested size; the guide's reference span is not a
published maximum or a promise of the author's refined output. The
[Refine](#refine) pass is the nearest the app has to that hires pass - the
same 1.5x and strength, with a plain resize in place of the author's
upscaling model.

The ungated Hugging Face entry is a community transport mirror, not the
author's repository or a LoRA. First load verifies the pinned 6.94 GB file's
SHA-256, converts it on CPU using pinned SDXL configuration, and caches the
full component tree with its embedded VAE and both text encoders. Allow about
14 GB for the original and conversion together. Later loads reuse that tree
with FP8, int8 or full bfloat16; this entry has no Nunchaku build. Saved
images retain the public mirror id. See the
[author's v17 card](https://tensor.art/models/990778216270553015/WAI-illustrious-v17),
[integration recipe](docs/design-notes.md#wai-illustrious-v17-an-author-checkpoint-through-a-pinned-mirror)
and [local verification](docs/verified.md).

For another anime model, choose **Z-Anime Base** in the menu or run:

```powershell
uv run python -m backend.main --model SeeSee21/Z-Anime --quant fp8 --offload model
```

Its defaults are 40 steps, CFG 4 and blank Mu. Use descriptive sentences and
a negative prompt; start at 1-1.5 MP, then try 1920x1088 (the 16:9, 2 MP
preset). The author's documented ceiling is 2048x2048. This experimental
Base model uses the supplied Diffusers FlowMatch Euler scheduler (shift 6),
which differs from the model card's ComfyUI Euler Ancestral/beta setup.
The first load downloads the pinned `diffusers/` tree and the standard Qwen3
encoder from the official Z-Image parent into the HF cache. This bypasses
Z-Anime's incorrectly packaged FP8 encoder; the distilled, GGUF and AIO
variants are not part of this selection.
See [the model card](https://huggingface.co/SeeSee21/Z-Anime) and
[local verification](docs/verified.md).

For **Anima Turbo v1.1**, choose it in the menu or run:

```powershell
uv run python -m backend.main --model circlestone-labs/Anima --quant fp8 --offload model
```

The app selects the full Turbo v1.1 checkpoint from the official repository,
with **8 steps and CFG 1**. Its authors recommend 8-12 steps and document
512x512 through 1536x1536 pixels. Tags and descriptive sentences can be mixed;
use lowercase tags with spaces, and prefix artist names with `@`. Negative
prompts have no effect at CFG 1. Mu, reference images and user LoRAs are
unavailable for this integration.

On first load, the app converts the pinned Turbo denoiser and text conditioner
on CPU, using the official Base Diffusers repository for shared components
and configuration. The cached tree lives under `$HF_HOME/local-diffusion/converted`
(normally `~/.cache/huggingface/local-diffusion/converted`) and is reused on
later loads. FP8, int8 and full bfloat16 (`--quant none`) use the normal loader;
this entry has no Nunchaku build. Status and saved images retain
`circlestone-labs/Anima`. See the
[model card](https://huggingface.co/circlestone-labs/Anima) and
[local verification](docs/verified.md).

For **NetaYume Lumina v4**, choose it in the menu or run:

```powershell
uv run python -m backend.main --model duongve/NetaYume-Lumina-Image-2.0 --quant fp8 --offload model
```

Start at 1 MP, 40 steps, CFG 5 and blank Mu. Write image tags or descriptive
sentences; the author's anime system instruction is applied automatically.
The published family guide recommends 40-50 steps and CFG 4-7. It still
describes v2, while this selection pins the author's v4 denoiser. The size
hint cites the base model's 1024-square example and the author's v2
1024x1536 workflow, not a measured v4 resolution limit.

The first load downloads the 5.22 GB v4 denoiser and the official Lumina 2
parent's shared Gemma 2 encoder and FLUX VAE, then converts on CPU into the
same cache used above. Original and converted weights both occupy disk.
Later loads reuse that tree with FP8, int8 or full bfloat16; Nunchaku is not
available for this entry. The fixed FlowMatch Euler scheduler uses shift 6;
Mu overrides that shift. This differs from the author's suggested ComfyUI
samplers. See the [model card](https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0)
and [local verification](docs/verified.md).

For **NewBie image Exp0.1**, choose it in the menu or run:

```powershell
uv run python -m backend.main --model NewBie-AI/NewBie-image-Exp0.1 --quant fp8 --offload model
```

Start at 1024x1024, 28 steps, CFG 4 and blank Mu. The step count and size
come from the model card; CFG 4 comes from the author's sampling script.
These are app starting settings. The card does not publish a minimum or
maximum resolution. NewBie accepts prose, tags and XML; for multiple
characters, try the author's XML prompt from Examples to keep each person's
appearance, clothing, action and position together.

The first load converts the pinned official denoiser, Gemma 3 encoder and
Jina CLIP text tower on CPU, caching the components alongside the other
converted models. Original and converted weights both occupy disk. The app
uses an empty system prefix, full-duration CFG with whole-image norm limiting,
and FlowMatch Euler shift 6; Mu replaces that shift. FP8, int8 and full
bfloat16 use the ordinary offload path. Reference images, user LoRAs and
Nunchaku are unavailable for this integration. See the
[model card](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1),
[integration recipe](docs/design-notes.md#newbie-exp01-pinned-components-and-an-explicit-sampling-recipe)
and [local verification](docs/verified.md).

A run whose stdin is not a terminal - a service, a script - never asks either.

Open <http://127.0.0.1:7860>. The first generation downloads ~32 GB of weights
for the default checkpoint (the transformer and the text encoder both) and
loads them — expect several minutes. Subsequent loads read from the HF cache.

`Ctrl`/`Cmd` + `Enter` generates from anywhere in the page, the prompt box
included - except from a box of the open Grid search fold, where it runs the
grid, and in the Refine dialog, where it runs the pass. With an image open,
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
`~/local-diffusion/.venv`, and checks that CUDA and the two extra packages are there.
It also says when the prompt critic's llama.cpp build cannot run there - glibc
older than 2.38, or `libgomp1`/`libssl3` missing, with the `sudo apt install`
line for those.
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
gallery (`LOCAL_DIFFUSION_OUTPUTS_DIR`, `LOCAL_DIFFUSION_DB_PATH`) and the HuggingFace cache on the
Linux filesystem — `/mnt/c` is a slow bridge and SQLite cannot lock across it —
and sets `LOCAL_DIFFUSION_QUANT=nunchaku` with a pre-quantized checkpoint:

| `LOCAL_DIFFUSION_MODEL_ID` | Card |
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
other `LOCAL_DIFFUSION_QUANT` value works from WSL too; set it before sourcing `env.sh`,
and `env.sh` then leaves `LOCAL_DIFFUSION_MODEL_ID` to the ordinary default rather than
pairing a torchao quantizer with an already-quantized repo (which downloads
everything and then fails inside transformers). `LOCAL_DIFFUSION_NUNCHAKU_FORMAT=int4`
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
cd /mnt/c/VSProjects/ImageGeneratorLocal
source wsl/env.sh
uv run python -m backend.clear
```

Git Bash is not a WSL shell: there `/mnt/c` does not exist, yet `env.sh` still
runs, and `uv` builds a second venv and an empty gallery under
`C:\Users\<you>\local-diffusion` and reports nothing to clear. The `database :`
line the tools print is the check — it should start with `/home/`.

`LOCAL_DIFFUSION_QUANT=nunchaku` implies `DIFFUSERS_TRUST_REMOTE_KERNELS=true`. The kernels
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

Leave `LOCAL_DIFFUSION_OFFLOAD` at `model`. With everything resident this checkpoint peaks
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
are the `LOCAL_DIFFUSION_*` variables under another name. `--dry-run` prints the
resolved settings without loading the model; `--quiet` drops everything but
one `<path>  seed <n>` line per saved image.

`--reference 12,15` conditions the run on gallery images by id (klein only;
see [Reference images and LoRA
adapters](#reference-images-and-lora-adapters)), and `--lora style.safetensors
--lora-scale 0.8` applies an adapter from `loras/`, with `--grid-lora-scale
0-1x0.25` sweeping the weight. Both are refused before the load, by name,
when the checkpoint or the installation cannot use them: a night that renders
every cell without the thing you asked for looks like a success.

`--refine 412` is the [Refine](#refine) pass: gallery image 412 scaled up and
re-denoised part of the way under the prompt and settings given, which with
`--from-image` are the image's own:

```powershell
uv run python -m backend.generate --from-image outputs/….png --refine 412 --scale 1.5
uv run python -m backend.generate --from-image outputs/….png --refine 412 --scale 2 --strength 0.3
uv run python -m backend.generate --from-image outputs/….png --refine 412 --scale 1.5 --grid-strength 0.2-0.5x0.1
```

With no size given the pass renders at the source's own size times `--scale`
(default 1.5, each side rounded to 16); `--width/--height` or `--aspect/--mp`
set another, and a shape that is not the source's crops it rather than
stretching it. `--from-image` supplies the recorded size, so add `--scale`
explicitly to enlarge it. `--strength` defaults to the checkpoint's own starting value
(0.4 on SDXL, 0.1 on FLUX.1 [dev]), and `--grid-strength` sweeps it. A pass
may go past the 4 MP presets, to any 2 MP preset at 2× (4448 on a side,
8.4 MP).
Refused before the load on a checkpoint that has no such pass, for a source
the gallery has not got, and for a strength that leaves less than one step.

`--from-image outputs/….png` starts from an image you already have: the
prompt, negative prompt, size, steps, CFG, true CFG, mu, seed, checkpoint,
references, adapter, and a refine pass's source and strength come out of the
PNG's own metadata, and any flag you give explicitly wins over it. The image's grid variables - a `{var}` value,
or a comparison version's name, which keep it on its row of a sheet - carry
too, but only while the prompt and negative prompt are the image's own (a
label on a changed prompt would name a version it is not), and a
`--grid-prompt` replaces them with its own. A comparison version's prompt
is taken verbatim, braces and all. The quantization is not carried - a WSL
Nunchaku PNG read on Windows would ask for a mode this machine cannot run -
and the tool says so when it differs, since the pixels then will too.

The CLI loads its own copy of the weights, so before it does it asks the
server on `LOCAL_DIFFUSION_HOST:LOCAL_DIFFUSION_PORT` whether one is already on the GPU. If the
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

Every command-line tool (`generate`, `clear`, `stats`, `reindex`) writes
UTF-8 when redirected to a file, a pipe or `> NUL`, where Windows would
otherwise use the locale's code page and stop at the first CJK or emoji
prompt; the Windows console is UTF-8 already, and a Linux terminal keeps its
own encoding. Windows PowerShell's `>` and `|` decode a program's output
with `[Console]::OutputEncoding` before writing it again, so for the file
to hold the text as written, set that to UTF-8 first
(`[Console]::OutputEncoding = [Text.Encoding]::UTF8`) or redirect from cmd
or Git Bash.

### Which settings did you keep?

```powershell
uv run python -m backend.stats                    # the newest run
uv run python -m backend.stats --run ab12cd       # one grid, however many jobs
uv run python -m backend.stats --by steps,cfg,prompt --min 3
uv run python -m backend.stats --all              # the whole gallery, indicative
uv run python -m backend.stats --job ab12cd --by variant --min 4   # a comparison, per version
```

The starred *rate* per setting - made, starred, percent - rather than the
count, which only rewards whichever setting you used most. The default scope
is the newest run, where the grid crossed its axes against the same prompts
and seeds and the comparison is fair; `--all` is confounded by whatever was in
use while the prompts happened to be good and says so. Settings with fewer
images than `--min` (5) are left out, so a lucky 1-of-1 does not lead. The
column says *starred*, not *kept*: nothing records a rejection, so an
untouched night reads 0%, not unknown. Rows rank by rate, then by count; a
tie keeps the order the settings were first made in. A prompt is folded onto
one line in the table and cut to 40 characters. Use `--job` for a comparison:
the same prompts and seeds can share a run even after their version labels
are swapped. If a label means different prompts within the same variable
assignment, `--by variant` asks for a narrower job or `--by variant,prompt`.

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
uv run python tools/fake_server.py --port 7882 --step 0.3
node tools/browser.mjs tools/smoke-jobs.json      # the queue, Pause, a grid's skipped cells, the load and the server gone
uv run python tools/fake_server.py --port 7865 --model baidu/ERNIE-Image-Turbo
node tools/browser.mjs tools/smoke-ernie.json     # the explicit prompt expansion editor flow
uv run python tools/fake_server.py --port 7866 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-animagine.json # tag guide, defaults, size range and disabled Mu
uv run python tools/fake_server.py --port 7878 --model cagliostrolab/animagine-xl-4.0
node tools/browser.mjs tools/smoke-tags.json      # chips, autocomplete, Tags and Examples by keyboard, Undo
uv run python tools/fake_server.py --port 7867 --model OnomaAIResearch/Illustrious-XL-v2.0
node tools/browser.mjs tools/smoke-illustrious.json # hybrid prompting, native size range and disabled Mu
uv run python tools/fake_server.py --port 7869 --model circlestone-labs/Anima
node tools/browser.mjs tools/smoke-anima.json    # Turbo defaults, size range, disabled Mu and saved settings
uv run python tools/fake_server.py --port 7870 --model KBlueLeaf/Kohaku-XL-Zeta
node tools/browser.mjs tools/smoke-kohaku.json   # hybrid tags, defaults, size guidance and saved settings
uv run python tools/fake_server.py --port 7872 --model NewBie-AI/NewBie-image-Exp0.1
node tools/browser.mjs tools/smoke-newbie.json   # XML examples, defaults, size guidance and saved settings
uv run python tools/fake_server.py --port 7873 --model LyliaEngine/waiIllustriousSDXL_v170
node tools/browser.mjs tools/smoke-wai.json      # WAI recipes, defaults, reference sizes and saved settings
uv run python tools/fake_server.py --port 7867 --fresh
node tools/browser.mjs tools/smoke-paging.json    # the gallery paging itself on scroll; needs the fresh gallery
uv run python tools/fake_server.py --port 7874 --model black-forest-labs/FLUX.1-Krea-dev
node tools/browser.mjs tools/smoke-analysis.json  # Check, the critic and Compare on a FLUX-shaped server
```

The fake server draws gradients stamped with seed and steps, so grid cells can
be told apart on a sheet, behaves as the claimed checkpoint's profile predicts
(the negative prompt's threshold, whether real CFG and Mu apply, and whether
prompt expansion is available), and
`--step 0.3` slows each denoising step enough to watch Pause, the queue and
the preview work. A cell whose prompt holds `[fail]` raises, as an
out-of-memory cell would, and `POST /api/fake/load` (`{"seconds": 4}`, or
`{"fail": "gated"}`) makes the loads after it slow or gated - what
`smoke-jobs.json` drives the skipped cells, the load's wait and a failed
load with. Its prompt check counts with stand-in tokenizers (a token
per word or mark, at each encoder's real budget) and its critic is a stand-in
that answers from the prompt after `--critic-delay` seconds (0.4 by default),
and refuses at once a prompt too long for its window - on the token
estimate, where the real one refuses at once only what cannot fit however it
tokenizes and the rest once its server has counted it, so near the window's
edge the two can differ;
`--no-critic` answers as a platform without one. `tools/browser.mjs` needs only Node 22+ and an installed Edge
or Chrome (`LOCAL_DIFFUSION_BROWSER` points at another executable); each run gets a
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
a side, 3168×1328 in pixels - and refused past it; only a [Refine](#refine)
pass goes further, to any 2 MP preset at 2×. The table in
`backend/sizes.py` is the one source, and the CLI, the browser and this README
are held to it by a test.

## Refine

**Refine** scales an image you have already made up and re-denoises it part
of the way, which keeps the picture. A direct render at the larger size does
not: the latents are drawn at the new shape, so the same seed is a new
sample, and past a checkpoint's documented range the composition is what
degrades - FLUX.1's own size advice is "render at 1-2 MP and upscale", and
WAI's author renders at 1024×1344 and then runs a 1.5× hires pass.

Press **⤢** on a gallery image, or **⤢ Refine** in the lightbox. A Refine
dialog opens on that image, the *source*, and the panel is left as it was:
Generate, Batch, Repeat and the grid always render from the prompt, and a pass
never takes their settings. The dialog starts from the image's own prompt,
sampling settings and seed, at 1.5× its size (or the last scale you ran)
unless that is past what the checkpoint holds, when it opens at the largest
factor that is - the one it skipped says so on its button, and running the
size it opened at leaves your scale as it was. On a wide window the source
is shown large beside the controls. **Strength** has focus; **Refine**,
Enter in any box or Ctrl+Enter runs the pass - once the server has read
what the boxes hold, so a strength it refuses is said in the dialog - and
its progress shows in the panel as any job's does. The **1× / 1.25× / 1.5× / 2×**
buttons multiply the source's size, shown under each - 1× re-denoises at the
size it has - and any other size works too: with **Keep shape** ticked,
typing W or H moves the other to the source's shape, and a shape that is not
the source's crops it rather than stretching it, which the size line
quantifies and the large view frames. **Versions**
makes 1, 2 or 4 passes on consecutive seeds from the **Seed** box, which
starts at the source's. The prompt, negative prompt, steps, CFG and Mu are
folded under **Prompt and sampling**: edit the prompt there if the detail
should change. An image made on another checkpoint keeps its prompt and
seed, and takes this checkpoint's steps, CFG, Mu and strength, with the fold
open to say so. Under the boxes the dialog says how many images the pass
makes, about how many steps run and, once the gallery holds passes at that
size, how long it will take; above that, where the encoders stop reading a
long prompt (on SDXL, 77 tokens - a tag added at the end of a long prompt
does nothing), with **Show** and the prompt check's **Apply**. The passes
already made from the image are a row at the top, each opening in the
lightbox beside the source. Closing the dialog without running keeps what
you typed for the next time you open the same image; **Start over** drops it.

A pass may go past the 4 MP presets, to any 2 MP preset at 2× - 4448 on a
side, 8.4 MP - because it takes its composition from the source, which is
what a direct render at that size loses. That is a guard against typos, not
a promise that it fits: measured here, WAI reached 8.6 MP in 46 s with
the card two-thirds used, while Z-Anime filled the 16 GB at 4.2 MP and took
12 minutes at 8.4 MP, and FLUX.1 Krea [dev] fills it already at 2.4 MP. Past
that point the driver spills into system memory and every step slows many
times over. The dialog's size line warns past the largest pass measured,
naming the run it quotes - FLUX.1 [dev] and the 4-bit builds were not
measured, and quote Krea [dev]'s fp8 run as that - and past 1600×1600
where nothing was.

**Strength** is how much is redrawn. The source is resized, encoded, noised
to that point on the schedule and denoised from there: low values keep the
picture and sharpen it, high ones repaint it, and 1 keeps nothing. Steps is
still the whole schedule, of which about Steps × Strength run, so a pass
costs less than a render at that size.

The box starts at the checkpoint's own value, like Steps and CFG, because a
strength is an index into the checkpoint's schedule and the same number does
different things on different ones: **0.4** on the SDXL checkpoints (WAI's
author documents 0.35-0.5 for the hires pass), **0.1** on FLUX.1 Krea [dev]
and [dev], **0.15** on Z-Anime, 0.25 on Z-Image, 0.3 on Qwen-Image and 0.5 on
the 4-step schnell. Each starts the pass at about half noise for a 1.5× pass
of a 1 MP image. On FLUX.1 that is a small number because its schedule's
shift grows with the size: measured here, 0.4 at 1.5× keeps the composition
and repaints the face, and at 2× even 0.1 starts at sigma 0.75 and repaints
the man. A Mu set for the pass moves it back (Mu 1.15 at strength 0.2
starts where the 1.5× default 0.1 does, with twice the steps, and at 2×
kept him), and the dialog says so under Strength once a FLUX.1 pass is past
the 1536×1536 measured. Several strengths typed into the dialog's **Strength** box -
`0.05,0.1,0.2` or `0.05-0.2x0.05` - run as one sweep over the source, which
opens as a sheet of how much of it each value kept; **Compare 3** writes the
checkpoint's own value and either side of it (0.05,0.1,0.15 on FLUX.1), and
the arrow keys step a single value by 0.05.

It runs the image-to-image pipeline diffusers ships beside the checkpoint's
own, over the weights already loaded, with no extra weight download or copy.
Under model offload the source's VAE encode leaves the VAE on the card beside
the denoiser until the pass finishes; measured WAI and FLUX.1 peaks are about
0.16 GiB above direct renders at the same size. The pass is available where
this app's pipeline has a compatible image-to-image counterpart:

| Refine pass | Checkpoints |
| --- | --- |
| yes | FLUX.1 Krea [dev], FLUX.1 [dev] and [schnell], Animagine, Kohaku, Illustrious, WAI, Z-Image Turbo, Z-Anime, Qwen-Image |
| no | Krea 2, FLUX.2 [klein], ERNIE-Image, NetaYume Lumina, Anima Turbo, NewBie |

Anima Turbo uses this app's custom pipeline, which has no compatible sibling;
diffusers itself does provide an Anima modular image-to-image workflow.

Elsewhere ⤢ says so instead of opening the dialog, and a request is refused
at submit rather than rendered as something else.

The source and the strength are part of the image's record - the row, the
PNG, the resume key - so **Reuse** on a refined image opens the dialog on
its source with the pass as it ran, `--from-image` reproduces a pass,
**Re-render** carries it to another size, the lightbox reads
`refine of #412 @ 0.4`, and `refine:412` in the search box finds every pass
made from image 412. The source is a gallery id, like a reference image:
delete it and its refinements stay, but cannot be made again.

It is not the WAI author's exact recipe, which upscales with R-ESRGAN 4x+
Anime6B before its hires steps: the resize here is a plain Lanczos one and
the detail comes from the re-denoise. From the terminal it is `--refine`
(see [Headless](#headless)).

Measured on the RTX 5080 (fp8, model offload, 2026-10-03; the rest is in
[verified.md](docs/verified.md#refine-pass)):

| Checkpoint | Pass at its starting strength | Time | A direct render at that size |
| --- | --- | --- | --- |
| WAI Illustrious v17 | 1024×1344 → 1536×2016 at 0.4, 11 of 28 steps | 12 s, 3.9 GiB peak | 23 s, and a different picture |
| FLUX.1 Krea [dev] | 1024×1024 → 1536×1536 at 0.1, 3 of 28 steps | 17 s, 12.5 GiB peak | 101 s, and a different picture |
| Z-Anime Base | 1024×1024 → 1536×1536 at 0.15, 6 of 40 steps | 33 s, 8.1 GiB peak | not run |

On all three the same pass twice is the same pixels, image 2 of a batch is
exactly what its own seed makes alone, a cancel lands within a step, and the
checkpoint's ordinary renders are byte-identical before and after. Versions
run one pass per image, so they cost time rather than VRAM, and the batch
pixel guard below does not apply to them. The passes past 4 MP are in
[verified.md](docs/verified.md#past-4-mp). The other three
SDXL checkpoints, FLUX.1 [dev] and [schnell], Z-Image Turbo and Qwen-Image
have not been run here: the SDXL ones share WAI's pipeline and starting
strength, FLUX.1 [dev] carries over Krea [dev]'s measurement on the same
schedule, and schnell, Z-Image Turbo and Qwen-Image start from schedule
arithmetic alone. The pass
has not been tried under Nunchaku in WSL.

## Batch vs Repeat

**Batch** produces N images in a single pipeline call, all resident in VRAM
together. It is the fastest way to get a handful of 1024s and the reason a
large 2K batch does not fit. The default guard allows every 1 MP preset at
Batch 12 and two 4 MP images; because real usage varies by checkpoint and
quantizer, `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` can override it.

**Repeat** produces N batches one pipeline call after another, stepping the
seed by the batch size each time, so a sweep costs no more VRAM than a single
batch. That is what makes an unattended run of large images possible: set
Repeat to 12 at 2048×2048 and walk away. Leave the seed blank and the sweep
starts from a random one; set it and the whole sweep is reproducible. The line
under the boxes spells it out ("6 images: 3 batches of 2, one after another,
seed +2 each, from a random start."). It is the request's `runs` field and the
CLI's `--runs`; the panel calls it Repeat because a *run* is a submission -
what the gallery's Runs picker lists, `run:` searches for and the last-run
banner reports - and one Repeat-3 Generate is one run of three batches.

**Cancel** appears over the progress bars while a job is running, beside
Pause. It stops at the end of the current step — within a second or two — and
keeps every image the run had already finished. The line it sits on names the
job it stops ("Now: a fox in the snow · 1024×1024 · 3 images", or "Now: grid ·
a fox, {mood} · …" for a grid), since with a queue the prompt box may hold
another job's text; hover it for the whole line where the panel cuts it.

**Live preview**: while a cell denoises, a small picture of it appears in the
viewer from a few steps in - a linear read of the latents the step callback
already carries, blurry but honest about the composition, refreshed a few
times a second and replaced by the real image when it lands. On FLUX.1 and
Krea 2; FLUX.2 [klein]'s pipeline gives the callback no way to unpack its
latents, so it shows none. With Cancel landing within a step, a prompt that
is going wrong costs five seconds rather than the render.

**Pause** holds the run at the end of the current cell and hands the GPU back:
between cells the weights are off the card under the default offload, and what
it still holds is the allocator's cache, which is returned. The button says
*Pausing…* at once - pressed again, it withdraws the request - and *Resume*
once the worker has actually stopped — until then a denoise may still be
running — and resuming costs nothing but the next cell's own transfers. A job
of one image (or one batch) has no boundary to stop at, so its Pause is held
and says so. The weights stay loaded, so Unload is still refused while a job is
paused.

**Queueing**: Generate and Run grid stay live while a job runs. A second press
queues the job behind the current one on the server; under the bars a row per
waiting job names it - its prompt, or the image a refine pass is of; "Next:"
where only one waits - with an × that cancels that one (and *Cancel queued*
for all, where there are several),
and the page moves on to the next job when the watched one ends. A reload
attaches to whichever job holds the GPU, then to the oldest one waiting.

**Away from the panel**: the tab's title follows the job ("54% · image 2/4",
"⏸ Paused", "✓ Finished 4 images" until you look at the tab again), and the
top bar carries the same counters in one line with a thin progress line along
its foot. A finished job's images join the head of the gallery without
moving it, so browsing old images during a wait keeps your place; when the
first image lands in the viewer while it is scrolled out of sight, *New image ↑*
beside the progress goes up to it - the page never scrolls on its own.

**When the server stops answering** the top bar's dot turns red and reads
"server not answering — retrying", Generate, Run grid and Unload are held with
that reason, and a job being watched says how many of its images were saved rather
than promising it carries on. Once the server answers again the messages go;
if it was restarted, the page says when, that the model is not loaded, and
that the job that was running ended with it (a grid's Resume finishes it).

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
prints the same warning). One grid is capped at `LOCAL_DIFFUSION_MAX_GRID_IMAGES` images
and each axis at 64 values, because `1-1000` is eight characters. A range
without a step counts in ones, so one that would stop at its first value -
`0.8-1.2` on Mu - is refused with a request for the step (`0.8-1.2x0.1`).

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
read a file the app did not write. Each chip in the panel carries the number a
prompt names it by - "the jacket from image 1" - and the order is what the
numbers mean: drag a chip, or press Alt+Left/Right on its ×, to move it, and
Delete on the × drops it. A chip whose image has since been deleted says so.

They cost less than you would think. Attention is quadratic in the sequence
length and each reference roughly doubles it, but measured here (klein 4B,
fp8, model offload, 1 MP, 4 steps) one reference ran in 8.2–9.2 s against
10.2 s for the same prompt without one, and two in 11.3 s: under model offload
the per-image weight transfer dominates, and the longer sequence disappears
into it.

**LoRA adapters** are read from `loras/` — drop a `.safetensors` file in and it
appears in the select (`LOCAL_DIFFUSION_LORAS_DIR` moves the directory). One adapter at a
weight, swapped on the live pipeline between cells, which is what makes
`0-1x0.25` an ordinary sweep rather than five reloads: the weights load once
and each cell is a re-weighting. Both the adapter and the weight are recorded
on the row and in the PNG, so Reuse restores them and a resumed grid tells a
0.5 cell from a 0.75 one. Not available under `LOCAL_DIFFUSION_QUANT=nunchaku`: those
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
other way makes two cells of nonsense. The grid preview under the fold shows the
prompts the block resolved to, and marks a variable left with one value - most
often `size = big, small`, commas where the `|` goes.

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
that. Once the gallery has timed renders at the grid's size, the grid preview
says how many of the seeds the minutes complete, and that the load or a job
queued ahead spends them too while there is one. The stop is reported as the
budget working, not as a cancel, and Resume finishes what was not reached.

**Seed list** is the second night: the two seeds that gave the composition you
want, crossed with a finer sweep. `4471, 8823` or `100-103` runs the whole grid
at exactly those seeds, in place of the count and the Seed box (`--seed-list`
in the terminal), and a resume of such a grid needs no base seed - the list is
the record.

While a job runs, the line under the progress bars says how long it has been
going and roughly how long is left, from the job's own pace so far - which is
what shows a run that has slowed down. When it finishes, a line under the
progress says how many images it made and in how long - and how many of them
were distinct pictures, when that is fewer - with what to do next: a plain
job's seed and *Use seed*, which puts it in the Seed box and nothing else;
*Show sheet* for a grid (never opened for you); and, where cells were skipped,
*Retry the N skipped cells*, which sends the grid as it was submitted again
with Resume and its base seed, whatever the panel holds now. That line stays
until the next job finishes - "Previous job:" while a queued one runs - so a
grid's Retry is still there in the morning with jobs queued behind it.
Skipped cells are counted on the running line as they happen and named, with
their seeds, at the end. A job waiting for
the weights says so, with the time it has waited and the checkpoint's name in
the top bar, rather than claiming to be behind another job; **Load** in the top
bar starts the download without queueing an image to do it. A failed load
fills the empty viewer with what went wrong and what to do - for a gated repo,
the model page to accept its terms on and `uv run hf auth login` - and **Retry
load**, which is also the top bar's button then.

**Resume** skips the cells the gallery already holds: tick it and re-submit the
same grid, and a night that died at cell 300 of 480 costs the cell it died on,
not the run. Left unticked, the grid preview says how many of its cells are
already there. It matches on the settings that make the image rather than on the job
that ran them, so a *new* job continues a dead one and an image rendered by hand
earlier counts too. Each prompt of a prompt grid counts on its own - finishing
the big dog does not skip the small one. It needs the
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
refine:412        the refine passes of gallery image 412; refine:none for the direct renders
strength:0.3-0.5  how far a refine pass was re-denoised
date:today        a day on your clock: today, yesterday or 2026-09-07
a lighthouse      anything left over searches the prompt
```

`date:` is a local day, not a UTC one: an overnight run crosses midnight in
every timezone, and the day you mean is the one on your calendar.

The gallery shows the images made on the checkpoint the server is running,
and nothing else, until you tick **All models** beside Favorites. Steps, CFG
and mu are the checkpoint's, so a picture from another one is a set of
settings you cannot reuse under a Reuse button that warns; hiding them is
the default, not a filter you have to remember. The tick says what it would
add - `All models (+40)` - counted under the same search and Favorites, so a
run picked or typed that was made on the other model reads as forty hidden
rather than as an empty grid. It is not remembered across a reload, and the
Runs picker, Re-render and Delete unstarred all follow it: with it off, none
of them can reach an image the grid is not showing. Scripts get the same
default from `GET /api/gallery` and pass `all_models=true` for everything.

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
the size set in the panel, one cell per image with its own prompt, negative
prompt, steps, CFG, true CFG, mu, references, adapter and weight, seed and
variables, as one job - the variables being a prompt grid's values or a
comparison's version, so the re-rendered run's sheet keeps its rows. Star the
winners of a 1 MP night, tick Favorites, pick 4 MP, click. Cells already
rendered at that size are skipped, so it can be resumed like a grid, and
a selection made on another checkpoint - which only All models lets in -
is refused rather than run with the wrong steps and CFG (untick it, or
let `model:` narrow the selection). Be honest with yourself about what
comes back: the pipeline draws its noise at the new shape, so the same seed
at 4 MP is a new sample - the prompt and the settings carry over, the exact
picture does not.

**Delete unstarred**, at the end of the same bar, is the other end of that
night: a run makes hundreds and earns a few stars, and the gesture you want
is "keep these", not "not this one" two hundred times. It removes every
image the filter matches that is not starred - `run:ab12cd` for the night,
the box empty for the whole gallery, or for this checkpoint's part of it
unless All models is ticked, which the dialog says - after a dialog that
shows the count, what stays and a few of the prompts, and asks for the count
typed back: the
mistake it guards against is the wrong *scope*, a run you have not looked
at yet, and the number is what gives that away. The delete is pinned to
the number, so images a running job lands while the dialog is open are
refused rather than swept, and the search has to be `key:value` terms or
empty - free text is matched fuzzily, and a deletion is not confirmed
blind. A starred image keeps what it was made from - a refine pass its
source, a klein render its reference images - and the dialog names those
too, with how many runs the delete reaches into and when they were made.
It is `clear --where ... --keep-favorites` (below) over HTTP: rows
and their files, no orphan sweep.

**Runs**, the picker beside the search box, lists the gallery one line per run
- when it started, how many images, how many starred, the prompt - newest
first; picking one searches for it and opens the sheet. And a page opened
after a run has ended says so under the panel: when it finished or stopped,
how many images and cells it reached, and the error if there was one - the
server keeps that for as long as it is up, and the note is shown once.

**Sheet**, above the gallery, lays that run out as the grid that made it: the
two widest-varying axes become the columns and rows, and a third becomes a
selector rather than a dimension nobody can read. The one exception: a
comparison's versions (the `variant` axis, and so also a variable a prompt
grid of your own names `variant`) always take the rows when anything else
varies too, one row per version. A variable's values keep the order the grid
declared them in - `size = small | big` reads small, big - and with
Favorites ticked, or a search narrower than the run, the sheet asks for the
whole run once to keep that order; after Delete unstarred, a value whose
first images were deleted can move down. Cells take the run's own
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
unreadable. Where it would still leave labels alike or longer than a line -
successive edits of one prompt, searched by a word they share - each column
is labelled by its edit against the oldest (`+, film grain`, `−golden +blue`),
the columns in the order the prompts were written; a heading's tooltip is its
whole prompt. A comparison's rows read Original, Reordered, Rewrite and Your
version, each with its edit against the original under the name.

A sheet the page opens - a comparison, a strength sweep, a run from **Runs**
or a card's ⊞ - scrolls into view under the gallery's search bar, and its
cells are sized to fit the screen's height as well as its width, so a 4 x 3
is one screen; a matrix too tall even at the smallest cells scrolls in a box
whose headings stay in view. A run where only the seed varies wraps into
captioned cells rather than one long row. A line above the grid names what
every image shares (**Fixed:** steps, CFG, seed…) - with Favorites ticked,
what the keepers have in common. **F** on a focused cell stars it. A strength
sweep shows its source as the first column, and a refine pass carries its
strength on the cell where strength is not an axis. With and without
reference images is an axis of its own on klein, and images that share a cell
are counted in the note: repeats of the shown image (a grid run twice), or
images that differ only in something the sheet does not lay out (size,
checkpoint). **Sheet** over an empty search opens the newest run (clear the
search for the whole gallery), and turning it off puts back the search a
page-opened sheet replaced. On a run's sheet, **Resume this run** puts the
seed the run started from and its size in the panel, ticks Resume and
previews what is left - the grid in the panel has to be the one that made it,
and the preview says when it is not. The exported PNG also carries the prompt, the variables as declared,
the shared settings and when the run was made, and is named by its pins.

## Settings

All optional, set as environment variables:

| Variable | Default | Notes |
| --- | --- | --- |
| `LOCAL_DIFFUSION_MODEL_ID` | `black-forest-labs/FLUX.1-Krea-dev` | A catalogue id such as `krea/Krea-2-Turbo` or `cagliostrolab/animagine-xl-4.0` |
| `LOCAL_DIFFUSION_QUANT` | `fp8` | `int8`; `none` for full bf16 (~25 GB VRAM); `nunchaku` for a pre-quantized Nunchaku Lite repo (Linux only, see above) |
| `LOCAL_DIFFUSION_OFFLOAD` | `model` | `sequential` (lower VRAM, slower) or `none` |
| `LOCAL_DIFFUSION_OUTPUTS_DIR` | `outputs/` in the repo | Where PNGs are written. `wsl/env.sh` moves it onto the Linux filesystem |
| `LOCAL_DIFFUSION_DB_PATH` | `local-diffusion.db` in the repo | The gallery index. Keep it off `/mnt/c` under WSL: SQLite cannot lock across the bridge |
| `LOCAL_DIFFUSION_LORAS_DIR` | `loras/` in the repo | Where `.safetensors` adapters are read from. A request names one by file name; nothing joins a path from a request onto it |
| `LOCAL_DIFFUSION_MAX_BATCH_PIXELS` | `12902400` | Total pixels allowed in one pipeline call; covers Batch 12 across the 1 MP preset row and two images across the 4 MP row |
| `LOCAL_DIFFUSION_MAX_GRID_IMAGES` | `1024` | The most images one grid search may queue — a guard rail against a typo becoming a week of GPU time |
| `LOCAL_DIFFUSION_PRELOAD` | `false` | Load weights at startup instead of on first request |
| `LOCAL_DIFFUSION_DEVICE` | `cuda` | `cuda:1` puts the weights on the second card, offload included, and the status line reports that card |
| `LOCAL_DIFFUSION_HOST` | `127.0.0.1` | |
| `LOCAL_DIFFUSION_PORT` | `7860` | |
| `LOCAL_DIFFUSION_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Host headers the API answers to, and with the port, the origins allowed to write to it. Needed when `LOCAL_DIFFUSION_HOST` is `0.0.0.0`: the browser sends the machine's name or LAN address, never `0.0.0.0` |
| `LOCAL_DIFFUSION_CRITIC` | `true` | `false` turns the prompt critic off: no Ask the critic, and nothing downloaded for it. Check and Compare stay |
| `LOCAL_DIFFUSION_CRITIC_SERVER` | unset | Path to a `llama-server` of your own, used instead of downloading the pinned llama.cpp b11206 build - for a platform with no pinned build, a Linux whose glibc is older than the pinned build's 2.38 (Ubuntu 22.04, Debian 12), or a newer release. Not held to the pin, but it must take b11206's flags (`--no-ui --offline --fit on --reasoning-budget --sleep-idle-seconds`), read its key from `LLAMA_API_KEY` and keep `/v1/models` behind it, or every critique fails. Name the binary itself rather than a script or launcher that starts it: a wrapper is stopped with everything it starts, but a server it moves into a session of its own escapes the stop, and on Windows and Linux its port is refused |
| `LOCAL_DIFFUSION_CRITIC_MODEL` | unset | Path to a `.gguf` used instead of downloading the pinned Qwen3-4B Q8_0. Not held to the pin. A symlink (a Hugging Face snapshot entry, say) is fine: answers are labelled with the name given, not the target's. Under WSL avoid `/mnt/c`: the critic's server reads the whole file across the 9P bridge each time it starts, which added about 20 s to every critique measured. The default location, the Hugging Face cache on ext4, added about 5 s once the file had left the page cache and nothing while it was cached |

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
and it fills the box (Ctrl+Z, or the Undo the note under the box offers, brings
back a prompt it replaced). Tag-caption checkpoints can also show **Tags**; for
Animagine it exposes only the model card's person-count, rating, quality, score
and year controls, inserts at the prompt caret - it opens under the box and
says where that is - finishes with the recommended
quality suffix, and can fill the recommended negative prompt. For Illustrious
it uses that model's documented person-count, quality-tier and year-modifier
terms and fills the authors' own negative prompt, without borrowing
Animagine's recipe; its prompting guide says up front that the checkpoint is
an untuned base with no default style, so the style has to come from an
artist or style tag. WAI v17 has its own quality suffix, negative prompt and
rating buttons. Its author publishes no vocabulary cutoff, so WAI's
autocomplete uses the live index without claiming those tags were learned by
the checkpoint. While typing a supported checkpoint's
tag prompt, autocomplete comes from Danbooru's live tag index (Tab or Enter
takes the first suggestion once three characters match it) and each
comma-separated item gets a green recognised, red unknown, or grey unchecked
chip; a green chip's edge is its Danbooru category, and the chips past where
the CLIP encoders stop reading are dimmed, with the token count beside the
tray's label. Illustrious remains hybrid: spellcheck stays on and a single natural-
language sentence does not become one unknown tag chip. On a wide page the
full validation list uses the results column, so it
does not move Generate and never gets its own scrollbar. Drag chips to reorder
the canonical comma-separated prompt (or Alt+Left/Right on a focused chip);
use each × or the Delete key to remove one. Click or right-click a gap to
insert at that boundary, right-click a chip to insert after it, or use
**+ tag** at the end, with inline autocomplete; Ctrl+Z undoes any of it. The
lookup is advisory and never blocks generation; if the network is
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
| FLUX.1 Krea [dev] | Descriptive sentences, subject first; camera settings optional, where framing matters; text in quotes | CLIP 77 + T5 512 tokens | 1024-1280 px sides recommended (1-1.6 MP); 0.1-2 MP documented |
| FLUX.1 [dev] | The same, 30-80 words; no negative prompt without True CFG | CLIP 77 + T5 512 | 0.1-2 MP documented; above that users report doubling |
| FLUX.1 [schnell] | The same; no negative prompt without True CFG | CLIP 77 + T5 512 in this app (BFL's reference stops at 256) | 0.1-2 MP documented |
| FLUX.2 [klein] | Subject, action, style, context; every rendered word in quotes; no negative prompt at all | Qwen3, 512 | 1024x1024 in every example; up to 2 MP recommended, 4 MP max |
| Krea 2 Turbo | One natural paragraph, long and detailed best, short works; text in quotes | Qwen3-VL, 512 | 1k-2k documented (1-4 MP); pretraining topped out at 1024 px |
| Krea 2 Raw | The same; an undistilled base Krea does not recommend for inference | Qwen3-VL, 512 | Trained to 1k: 1 MP |
| Z-Image Turbo | Long, plain, objective description, English or Chinese; text in quotes | Qwen3-4B, 512 | 0.26-4.2 MP documented, trained to the 1k-1.5k range |
| Qwen-Image | Dense caption in colour/shape/texture/quantity/text terms; append ", Ultra HD, 4K, cinematic composition."; True CFG 4 with a negative prompt | Qwen2.5-VL, 512 | Trained to 1328x1328; the README's shapes are 1.5-1.8 MP |
| ERNIE-Image Turbo | Expand a short prompt, then review its detailed Chinese paragraph and local English translation | Ministral-3, 2048 | Seven 1 MP shapes listed, nothing above |
| Animagine XL 4.0 Opt | Danbooru tags: person count, character and series, rating, subject/pose/scene, then quality tags; use its negative prompt | CLIP + CLIP, 77 tokens | Trained at 1024x1024; nine recommended shapes around 1 MP |
| Kohaku XL Zeta | Tags, natural-language descriptions or both; its own quality, rating and date tags | CLIP + CLIP, 77 tokens in this app | 1024x1024 or similar pixel count recommended |
| Illustrious XL v2.0-STABLE | Tags or natural-language descriptions; an untuned base with no default style, so name an artist or style, one composition tag, and use its negative prompt | CLIP + CLIP, 77 tokens | Native 1-2.25 MP documented; up to 4 MP with some loss |
| WAI Illustrious v17 | Tags or short descriptions; its own quality recipe and negative prompt | CLIP + CLIP, 77 tokens in this app | Above 1024x1024 recommended; 1024x1344 examples precede a separate hires pass; no published maximum |
| Anima Turbo v1.1 | Tags, descriptive sentences or both; lowercase tags with spaces, `@` before artist names; no negative prompt effect at CFG 1 | Qwen3-0.6B + text conditioner, 512 in this app | 512x512-1536x1536 documented (0.26-2.36 MP) |
| NewBie image Exp0.1 | Prose, tags or XML; separate character blocks for multiple people | Gemma 3, 512 tokens + Jina CLIP v2, 2048 tokens in this app | 1024x1024 example; no published resolution range |

### Checking a prompt

**Check**, in the Prompt caption between Tags and Expand, holds the prompt
against the running checkpoint - nothing is generated and no model is
loaded, so it answers at once, on the CPU, even on a cold server or beside a
running night. It reads the prompt, the negative prompt and the CFG / True
CFG that decide whether that negative runs - and, on klein, the reference
images - exactly as Generate would send them. The first check on a
checkpoint whose tokenizers are not cached yet downloads those tokenizer
files (megabytes, never weights). A check waits up to 30 seconds for them;
past that, the line under the Encoders rows says which tokenizer is still
loading, and the download carries on for the next check. Unload never waits
for it. On a tag checkpoint - and on Anima, whose "@ before an artist name"
needs to know which tags are artists - it asks Danbooru's tag index what the
comma items are, sending the tag text only, as the chips do; offline, those
checks stand down rather than guess, and the panel says they did (on Anima,
that not every tag was checked for an artist name). The panel it opens -
above the results, or beside the image in a column of its own on a window
1800 px wide or more - has:

- **Encoders**: one row per text encoder, tokens used against what the
  pipeline keeps, and - past that - "stops reading at ..." (click it to
  select the words that encoder never sees). Every pipeline here cuts an
  over-long prompt silently. On FLUX.1 the CLIP-L cut is only a caution: that
  encoder feeds one pooled vector and T5 reads on.
- **Order**: the order the checkpoint's guide states, when it states one
  (FLUX.1: subject first - on [dev], subject, setting, style, camera,
  lighting; FLUX.2 klein: subject, action, style, context; the tag
  checkpoints' tag orders; NewBie's XML blocks), with a link to the source.
  Only a tag checkpoint's comma tags are actually placed in it: on a tag
  checkpoint whose tags are out of that order, **Use reordered** moves the
  same tags, spelled the same, into it. A quotation or other sign text moves
  whole, commas and all, and is never read as a tag. Where the order is one
  of sentences (FLUX.1, klein), for XML or JSON, and for a tag checkpoint's
  prompt split by line breaks or fullwidth commas (，) rather than commas,
  the order is shown with "Not checked", and the summary says
  "order not checked" rather than letting "nothing to flag" read as a pass -
  ask the critic, or write your own reordering under Compare. Where Check
  could not place some tag, a placed order says "Partly checked", naming
  those tags and why, and the summary says "order partly checked": a tag
  with no Danbooru category is not placed, so one out of order may go
  unflagged. That is a tag the index has no usable category for (an OC's
  name, a trigger word, a typo, or a tag deprecated or newer than the
  checkpoint), a tag it was never asked about (offline, no usable reply
  from it, or past the first 64 a check looks up), a clause of a prose
  prompt, which is never looked up - only a whole sentence is placed, with
  the general tags - and lettering or an @ name, which are never looked up
  either. It is said only where the missing category decides the order: a
  prompt of one tag or one clause, or an unknown tag every category would
  leave in order (between `1girl` and `masterpiece` on Animagine) or out of
  it (ahead of `1girl`, which is flagged), is checked, and so is such a tag
  written twice, whose copies cannot be out of order beside each other.
  Where the guide states no order, the panel says the app checks none, and
  shows any order the authors' own texts give (Anima's card, Neta Lumina's
  prompt book for NetaYume, ERNIE's and Qwen's rewriter instructions) as
  information, with its source.
- **Findings**, warnings first. A warning means the text will not do what it
  seems to: cut off, a "no ..." where no negative prompt runs, A1111-style
  `(word:1.2)`, `((word))` or `BREAK` (nothing here parses weighting; the
  encoder reads the brackets and numbers as text), a typed control token,
  NewBie XML that does not parse. Each finding says what it rests on:
  **Guide** (the checkpoint's guide - its words follow the finding when
  short, and are the badge's tooltip either way; click for the source), **Authors** (an author's example), **Encoder** (what the
  pipeline's own tokenizer does) or **Heuristic** (this app's own pattern
  check, not an author's rule). Click a finding to select its words;
  **Apply** replaces the field with the fixed text - quotes added, tags moved,
  the recommended suffix or negative prompt - and Ctrl+Z undoes it. Unquoted
  sign text is quoted whole ("Fish and Chips", "Welcome to Paradise"); where
  its end is a guess ("says fish and chips for sale", Chinese lettering that
  runs into 的 or past twelve characters, English `MADE IN JAPAN`, or
  lettering continuing through punctuation in `9:00`, `$9.99` or a URL,
  lowercase words after "reading" or "saying"), or where the words may
  describe the lettering rather than be it (after "with the words" or "the
  text", anything not in capitals: "Happy Birthday", "set in bold type"),
  the finding asks for the exact words in quotation marks and offers no
  Apply. Chinese sign text ends
  for certain at 字样 or at a count of its characters (写有欢迎字样 and
  写着欢迎两个字 quote 欢迎); 等 before 字样, or a verb that places the sign
  (挂在, 位于), ends it as a guess, unless the sign itself asks for that verb
  (请站在黄线外, 禁止停放在此). A negative prompt's Apply is dead, and says
  why, while the negative box is greyed at the current CFG.

Editing either prompt, changing whether CFG or True CFG applies the negative
prompt (even when an open grid keeps its editor enabled), or changing the
reference images dims the panel
("prompt changed since", or "references changed since") and disables its
buttons until **Check again**. A checkpoint change clears it.

**Ask the critic** appears where a critic can run: x86-64 Windows, x86-64
Linux with glibc 2.38 or newer (WSL's Ubuntu 24.04 and later; not Ubuntu
22.04 or Debian 12), or anywhere `LOCAL_DIFFUSION_CRITIC_SERVER` names a
`llama-server`. On Linux the pinned build also needs `libgomp1` and `libssl3`
from apt, and a fresh WSL Ubuntu has no `libgomp1`: without them the first
critique stops before downloading anything and gives the `sudo apt install`
line to run (`wsl/setup.sh` checks for them too). A local Qwen3-4B, run by
llama.cpp on the GPU, reads the prompt against the checkpoint's guide
sentences (not those where the app describes its own menus, buttons or
pipeline choices, or gives advice of its own) and short excerpts from its
authors' own texts - model cards, prompting guides, the prompt-rewriter
instructions the authors published - and answers with the prompt taken
apart (subject, action, setting, style, lighting, camera, text to render),
up to six issues each citing one numbered source, a rewrite, and, where an
order is documented, the prompt reordered. Hover over an issue's source for
the cited words, and click it for their page - for a guide sentence a rule
also cites, the page that rule's finding links. Words that are not the
checkpoint's authors' own - this app's translation, a family or parent
model's guide - say whose they are under the issue, and on a checkpoint
outside the catalogue the app's general advice is labelled General, not
Guide.

What the server checks before showing any of it: an issue whose citation
names no source it was given is dropped; a quote that is not in the prompt
as whole words loses its quote; an issue that repeats a rule finding
already listed (a negation, quotation marks, the tag order, weighting), in
its wording or by quoting the words the finding is about, is left out - one
on the negative prompt only when the issue says it is about the negative
prompt - and so is one that speaks to the prompt's own length where the
guide gives a range and the check counted it (not in XML or JSON, which it
holds to no range); when the count was in range, the note says so rather
than naming a finding. A quote that only sits next to a finding, or on
characters an encoder cannot read, does not count as repeating it, and a
length or a word count worded as a sign's, a lens's or a detail's ("add at
least 5 words for the lighting", "only 3 words of the prompt are about the
lighting", "the lighting has only 3 words", "add 5 words to match the
lighting", "keep your lighting description under 5 words") is not taken
for the prompt's length; "the description", "the full description" or "the
revised description" on its own is the prompt's. That is read from the
wording, so a phrasing unlike these can still be taken either way. An order
issue the documented order reverses ("lighting first", or "put the
lighting before the camera" where the order has the
camera first) is left out, and so are order issues that cannot all hold
(two parts each asked for first, or "X before Y" beside a wish for Y first
or for Y before X); an instruction in the negative ("avoid putting the
style before the subject") asks for nothing it names. A part is kept only
as the prompt's own words, and "text to render" only as words the prompt
asks to letter. A "reordered" version that adds, drops or changes a word -
or, in a JSON prompt, writes a number otherwise, 0.50 as 0.5 - is
discarded; one that survives is shown in your own words - their case and
spelling as you wrote them - in the critic's order. On a tag checkpoint
whose tags Check placed, only Check's own reordering survives - unless
Check could not place some tag where that decides the order (the tag index
could not be asked about it, or had no usable category for it - what the
Order section calls "Partly checked"):
then the critic's survives if it keeps your tags (or a prose prompt's
clauses) whole and Check's order rule finds nothing out of order in it, and
is set aside, saying what the rule found, if not. The **Use reordered** button's
tooltip says which kind survived. Retained tags keep your parenthesis escapes
(`\(` and `\)`); removing those alone does not count as a rewrite.
A rewrite that is your prompt unchanged is not shown; one wrapped whole in
quotation marks is unwrapped, one that puts
words the prompt does not have in quotation marks - lettering nobody asked
for - is dropped, and one that quotes the prompt's description, or adds a
handful of words the prompt does not have, is flagged. A rewrite or
reordering too long for a prompt is left out too, and anything left out is
said so under the answer - but for a reordering that moves nothing, which
has nothing to show, and some of what the answer's schema forbids (a seventh
issue, say), which only a `llama-server` of your own that ignores the schema
can send. The rewrite is then put through the same checks as the prompt,
listed under it, each marked "also in your prompt" or "new", with the
prompt's findings it no longer has after them - apart from those that rest
on the tag index (a tag's order, an artist's @), which the rewrite's check
does not ask, so they are listed as possibly still applying. **Changed from original**
gives the number of characters added and removed, the line under it the
first few changes; open it to see the marked text. If no rewrite survives,
the answer says **No rewrite offered** and explains anything set aside.

Character and series names are optional: the tag order describes where to
put them when you use them. Generic and original subjects do not need names,
and the critic is instructed to keep them generic.

What survives is still a 4B model's suggestion - it can misapply a source,
and it pads short prompts with scenery and mood however it is told not to -
so read the rewrite before **Use rewrite**. **Use rewrite** and
**Use reordered** replace the prompt (Ctrl+Z restores it), and put the
prompt they replace into Your version under Compare while that box is
empty, so **Run comparison** sets the two side by side. A critique takes
10-20 seconds and about 6 GB (5.5 GiB) of VRAM, then the critic's server is
stopped (`docs/verified.md` has the measurements). It waits for the image
queue - the button is held while a job is queued or running - and Generate
and Run grid are held from the click until it answers, including the
re-check it runs first on an edited prompt. The first use downloads
llama.cpp's official release b11206 (about 0.6 GB) and Qwen3-4B Q8_0
(4.3 GB) under the Hugging Face cache directory (`HF_HOME`), each checked
against a pinned sha256, before the critic asks for the GPU; later uses
start from disk. The line under the button names the model it runs, as the
answer's footer does, and what a first use would still download; with
`LOCAL_DIFFUSION_CRITIC_SERVER` or `LOCAL_DIFFUSION_CRITIC_MODEL` set it
promises no time and no download for what you supplied.
`LOCAL_DIFFUSION_CRITIC=false` turns it off; `LOCAL_DIFFUSION_CRITIC_SERVER`
and `LOCAL_DIFFUSION_CRITIC_MODEL` use a `llama-server` or `.gguf` already on
disk instead of the downloads (see Settings).

**Compare on the same seeds** answers what no guide can: does a version of
the prompt change *this* checkpoint's pictures? Tick the versions -
original, reordered (the tag reordering, or the critic's where it survived
its check), rewrite, and **Your version**, a box for a version of your own
(**Start from the prompt** copies the checked prompt in to rearrange). Only
a reordered version - Check's, the critic's, or yours - isolates word order:
the rewrite changes wording and often content too, and on a sentence prompt
Check has no reordering and the critic's rarely survives its check, so
there the order is tested by writing it under Your version. Pick 4 or 8
seeds per version and comparison **Batch** 1, 2 or 4 (default 1).
**Run comparison** queues one grid - one image per version per seed,
the panel's other settings, the Seed box's seed or a drawn one - and
opens that submission's sheet with one row per version, original first,
unticking Favorites if it was on: nothing in the new comparison is starred
yet, and the sheet's note says so. The main panel's **Batch** and **Repeat**
do not apply. Comparison Batch controls how many images run together,
without multiplying the seed count: two versions on four seeds still make
eight images; Batch 2 queues two batches per version. The line above
the button gives the image count, seeds per version, batch size and
time estimate first. 4 seeds is a first look: the seeds are shared, so the
rows are pairs, and even a change on all four could be chance
(p = 0.125); 8 can tell a change on every seed from chance. The prompt box
and the grid's variables are left as they were. Star what worked, and the
starred rate per version is

```powershell
uv run python -m backend.stats --job <id> --by variant --min 4   # <id>: the sheet's job: term
```

One row per version, by its label: `--by prompt` would cut each prompt short,
where versions that open with the same words read the same. Reopening the
whole run can include earlier comparisons with the labels swapped; if a
label then means different prompts, the sheet keeps Prompt as an axis so
those images can be inspected separately.

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
| Kohaku XL Zeta | 28 | 5.0 | Unavailable |
| Illustrious XL v2.0-STABLE | 28 | 5.0 | — |
| WAI Illustrious v17 | 28 | 6.0 | Unavailable |
| Anima Turbo v1.1 | 8 | 1.0 | Unavailable |
| NewBie image Exp0.1 | 28 | 4.0 | Blank; static shift 6 |

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
`LOCAL_DIFFUSION_MODEL_ID`. It is a 4-step model with an 8 GB Qwen3 text encoder, so under
`LOCAL_DIFFUSION_QUANT=fp8` the 4B fits with room to spare.

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

**Kohaku XL Zeta** is available in the start-up menu, or directly:

```powershell
uv run python -m backend.main --model KBlueLeaf/Kohaku-XL-Zeta --quant fp8 --offload model
```

Its [official Diffusers tree](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta)
loads through the normal SDXL path, keeping the supplied Euler scheduler.
The app starts at **28 steps, CFG 5 and 1024x1024**, within the author's
12-50-step / CFG 3.5-6.5 ranges. Tags, natural-language descriptions and mixed
prompts are supported; the Tags picker supplies Kohaku's quality, rating and
date vocabulary, with Danbooru suggestions filtered to 2024-07-10. The card
describes training with longer captions, but this app's stock SDXL encoders
read **77 tokens**; keep the important details early. Larger sizes remain
selectable with a recommendation warning. Mu, True CFG and reference images
are unavailable, and there is no Nunchaku mapping for this entry. See
[the verification record](docs/verified.md) for measured runtime support.

Illustrious XL v2.0-STABLE (`OnomaAIResearch/Illustrious-XL-v2.0`) is also
available in the start-up menu, or directly:

```powershell
uv run python -m backend.main --model OnomaAIResearch/Illustrious-XL-v2.0
```

The [official repository](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)
publishes a single 6.94 GB checkpoint. On first load the app downloads it and
converts it on CPU into a cached Diffusers tree, then uses the normal
FP8/int8/bfloat16 loading and offload path. Allow about **14 GB of disk space**
for the original and converted weights. The tree lives under
`$HF_HOME/local-diffusion/converted` (normally
`~/.cache/huggingface/local-diffusion/converted`); later loads reuse it.
The official model id remains in status, the database and PNG metadata.
Checkpoint and SDXL configuration revisions are pinned in
`backend/backends/single_file.py`; no SDXL base weights are downloaded.

Its [authors' report](https://arxiv.org/html/2409.19946v1) supports tags
and natural language, and documents native 1-2.25 MP, with some loss up to
4 MP. The direct 2048x2048 illustration is a v1.0 family example, not a
v2.0-STABLE measurement. The app starts at 28 steps and CFG 5, within the
authors' >20-step / CFG 5-7.5 guidance, and fixes Euler Ancestral in the
converted tree. Mu and reference-image input are unavailable; there is no
Nunchaku build for this entry. Its **Tags** picker follows the report's prompt
order, quality tiers and year modifiers, and the shared Danbooru autocomplete
uses the archived card's August 2024 dataset cutoff. Natural-language prompts
remain supported; Animagine's suffix and negative prompt are not copied over.

Verified on the RTX 5080 with FP8 and model offload: one 28-step/CFG-5 image
at 1024x1024 took 15 s, 1536x1536 took 18 s, and 2048x2048 took 35 s. The
1536 natural-language sample included its requested subject; the 2048 sample
omitted it. These are runtime checks, not a promise of equal quality at 4 MP.
See `docs/verified.md` for settings and remaining measurement gaps.

## Output

PNGs land in `outputs/` with generation parameters written into the PNG text
chunks (both an A1111-style `parameters` block and a `local-diffusion` JSON
blob - the prompt pair, size, steps, CFG, true CFG, mu, seed, checkpoint,
quantization, the job and run ids, the variable assignment and the pixel hash),
so the files stay self-describing outside the app. The reader also accepts the
pre-rename metadata key. `local-diffusion.db` indexes them for search.

## Clearing history

```powershell
uv run python -m backend.clear                   # preview, then confirm
uv run python -m backend.clear --dry-run         # report only, change nothing
uv run python -m backend.clear --keep-favorites  # spare starred images and their sources
uv run python -m backend.clear --yes             # no prompt
```

Prints what it is about to remove — records, files, and any PNG with no row
left pointing at it — and asks for confirmation before deleting anything, then
vacuums the database. This clears the Windows gallery; the WSL one is a
separate database, cleared by the same command from a WSL shell after
`source wsl/env.sh` (see [Run under WSL 2](#run-under-wsl-2-4-bit-nunchaku)). `--keep-favorites` spares starred images and the
images one of them was made from (a refine pass's source, a render's reference
images), but an orphaned file has no row to carry a star and goes either way. Safe to run while the
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
orphan sweep stays off this path on purpose. The gallery's **Delete unstarred**
button is this same path - `--where <the search box> --keep-favorites` - with
the typed count as its confirmation.

## Rebuilding the index

```powershell
uv run python -m backend.reindex                     # outputs/, in place
uv run python -m backend.reindex --from D:\wsl-outputs   # copy in, then index
uv run python -m backend.reindex --from D:\backup\outputs --database D:\backup\local-diffusion.db
uv run python -m backend.reindex --dry-run
```

Every PNG carries its whole reproduction set, so `local-diffusion.db` can be
rebuilt from the files: a lost or corrupted index, or the WSL gallery merged
into the Windows one. Files whose names the index already has are skipped,
which makes it idempotent; files without a Local Diffusion block (other tools'
PNGs) are named and skipped; a file from elsewhere is copied in under its own
name unless a different file already sits there. Two things do not come back:
the favourite flag, which lives only in the database, and the duration,
recorded as 0 so the time estimate ignores those rows. When the matching old
database is available, pass both `--from` and `--database` (or `--db`) instead.
PNG-only recovery remaps reference and refine sources using their recorded
filename, pixel hash and dimensions. Older PNGs without that source identity,
or sources that cannot be verified, lose those links (and a refine's strength)
and are counted as unresolved. Updated conditioning ids are also written into
the PNG metadata during an in-place rebuild, keeping `--from-image` safe.
The database merge preserves favorites and durations, remaps reference and refine ids into
the current gallery, and updates the copied PNG's embedded Local Diffusion
metadata to carry those new ids. Rows already present by filename are skipped,
so the database merge is safe to rerun too. Missing images, unreadable files,
unresolved references, and filename conflicts are reported and never overwrite
the current gallery.

## Licence

Model weights have their own licences; Krea 2 uses the **Krea 2 Community
License**. Anima uses the
[CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima#license),
which permits commercial use of generated images. For WAI, consult the
[author's model permissions](https://tensor.art/models/990778216270553015/WAI-illustrious-v17);
the community mirror's metadata is not authoritative licensing guidance. This application code
carries no such restriction.
