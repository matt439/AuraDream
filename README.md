# Krea Local

Run **Krea 2 Turbo** or **FLUX.1 Krea [dev]** locally — up to 4 megapixels on a
single consumer GPU. FastAPI + diffusers backend, vanilla-JS frontend, SQLite
gallery. FLUX is the default because its repo is click-through gated; Turbo
needs approval first.

![stack](https://img.shields.io/badge/torch-2.13%2Bcu130-informational)
![stack](https://img.shields.io/badge/diffusers-0.40.0-informational)

## Requirements

- NVIDIA GPU, **16 GB VRAM minimum** (float8 path). 24 GB+ can run `KREA_QUANT=none`.
- ~35 GB free disk per checkpoint, measured from the HF cache: FLUX.1 Krea
  [dev] is 32 GiB (transformer 23.8 GB, T5 9.5 GB), Krea 2 Turbo 34 GiB. The
  WSL Nunchaku Lite re-uploads are far smaller.
- Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

## One-time setup

The Krea checkpoints and FLUX.1 Krea [dev] are **gated** HuggingFace repos; the
four FLUX.2 [klein] repos are open and need no login (Apache 2.0, and the
cheapest thing here to try first - 4 steps, ~3 GiB peak). For the gated ones,
log in first:

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
controls do, so the prompt is the only required argument. `--model`, `--quant`
and `--offload` are the `KREA_*` variables under another name. `--dry-run`
prints the resolved settings without loading the model; `--quiet` drops
everything but one `<path>  seed <n>` line per saved image.

`--from-image outputs/….png` starts from an image you already have: the
prompt, negative prompt, size, steps, CFG, mu, seed and checkpoint come out of
the PNG's own metadata, and any flag you give explicitly wins over it. The
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
composition. A hand-typed size is accepted up to the largest preset - 2736 on
a side, 1776×2368 in pixels - and refused past it; the table in
`backend/sizes.py` is the one source, and the CLI, the browser and this README
are held to it by a test.

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

They cross: two sizes, two grounds, four step counts and two seeds is 32 images.
The panel previews what it would run before you commit to it — the resolved
axes, the finished prompts, the image count, and once the gallery holds anything
at that size to measure against, roughly how long. One grid is capped at
`KREA_MAX_GRID_IMAGES` images and each axis at 64 values, because `1-1000` is
eight characters.

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
per origin, and dropped without complaint where storage is blocked.

**Seed list** is the second night: the two seeds that gave the composition you
want, crossed with a finer sweep. `4471, 8823` or `100-103` runs the whole grid
at exactly those seeds, in place of the count and the Seed box (`--seed-list`
in the terminal), and a resume of such a grid needs no base seed - the list is
the record.

While a job runs, the line under the progress bars says how long it has been
going and roughly how long is left, from the job's own pace so far - which is
what shows a run that has slowed down. When it finishes, the same line says
how many images it made and in how long. A job waiting for the weights says
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
mu:none           the images that ran with no shift at all — not the same as mu:0
date:today        a day on your clock: today, yesterday or 2026-09-07
a lighthouse      anything left over searches the prompt
```

`date:` is a local day, not a UTC one: an overnight run crosses midnight in
every timezone, and the day you mean is the one on your calendar.

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
head. The Prompt axis is labelled
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
| `KREA_MODEL_ID` | `black-forest-labs/FLUX.1-Krea-dev` | `krea/Krea-2-Turbo`, or `krea/Krea-2-Raw` for the undistilled base (52 steps, CFG 3.5) |
| `KREA_QUANT` | `fp8` | `int8`; `none` for full bf16 (~25 GB VRAM); `nunchaku` for a pre-quantized Nunchaku Lite repo (Linux only, see above) |
| `KREA_OFFLOAD` | `model` | `sequential` (lower VRAM, slower) or `none` |
| `KREA_OUTPUTS_DIR` | `outputs/` in the repo | Where PNGs are written. `wsl/env.sh` moves it onto the Linux filesystem |
| `KREA_DB_PATH` | `krea.db` in the repo | The gallery index. Keep it off `/mnt/c` under WSL: SQLite cannot lock across the bridge |
| `KREA_MAX_GRID_IMAGES` | `1024` | The most images one grid search may queue — a guard rail against a typo becoming a week of GPU time |
| `KREA_PRELOAD` | `false` | Load weights at startup instead of on first request |
| `KREA_DEVICE` | `cuda` | `cuda:1` puts the weights on the second card, offload included, and the status line reports that card |
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
