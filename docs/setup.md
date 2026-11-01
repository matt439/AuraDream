# First-time setup

A step-by-step path from a fresh Windows 11 machine to a first image. The
[README](../README.md) has every launch option; this page covers only what a
first run needs, in order. Allow some time for the first image: a checkpoint
download is several GB, and the default is about 32 GiB.

For the 4-bit Nunchaku route under WSL 2, see
[Run under WSL 2](../README.md#run-under-wsl-2-4-bit-nunchaku) instead;
`wsl/setup.sh` does the driver check, uv and the install for you there.

There are two routes: the [easy install](#easy-install), which a script does
for you, and the manual steps from [step 1](#1-check-the-gpu-driver) on,
which do the same by hand. Read [Before you start](#before-you-start) either
way.

## Easy install

1. Get the code. Either download it from GitHub (**Code > Download ZIP**)
   and extract it, or clone it with Git. Put the folder somewhere short that
   OneDrive does not sync, such as `C:\AuraDream`, not in Documents or on
   the Desktop when those are synced.
2. Open the folder and double-click **`install.cmd`**. (Windows may warn
   that the file came from the internet; choose **Run**, or **More info >
   Run anyway**, only if you got it from the project's GitHub page.) A window opens
   and the installer explains each step as it goes. It needs no
   administrator rights, and it is safe to run again: it skips what is
   already done.

What it does, in order:

- Checks Windows, the folder (OneDrive, a very long path or non-English
  characters get a warning), the NVIDIA driver (CUDA 13.0 or higher, from
  `nvidia-smi`), the card's generation (compute capability 7.5 or newer)
  and video memory, and the RAM, which it compares with the measured
  figures below; it promises nothing for other machines. A driver that is
  too old or a card that is too old stops it with what to do.
- Installs **uv** and **Git** only if they are missing, with winget, or
  with uv's own installer (a pinned version) where winget is not available.
- On a ZIP download, offers to turn the folder into a Git checkout of the
  latest version so `update.cmd` works. Your images, gallery and settings
  are not code and are left alone; a code file that differs from the latest
  version is copied to a `zip-files-replaced-<date>` folder first. You can
  say no and carry on without updates.
- **Asks where model downloads go** (`HF_HOME`, [step 3](#3-choose-where-the-downloads-go))
  before anything downloads, showing each drive's free space; Enter keeps
  the default. It asks nothing if `HF_HOME` is already set. When the project
  is on another drive than uv's package cache, it offers to keep the cache
  on the project's drive (`UV_CACHE_DIR`) so the packages are not stored
  twice.
- Runs `uv sync` (a few GB of packages, never model weights), checks that
  PyTorch sees the GPU, and runs the read-only machine check
  `uv run python -m backend.doctor`, which you can also run yourself at any
  time.
- Only if you want the gated default checkpoint, opens the Hugging Face
  account, licence and token pages and runs `uv run hf auth login`, where
  you paste the token into Hugging Face's own tool (the installer never sees
  it). Otherwise it recommends starting on an open checkpoint, Animagine
  XL 4.0.
- Makes a Start-menu shortcut and, if you agree, a desktop shortcut named
  **AuraDream**. Each runs `bin\auradream.cmd --open`: a window with the
  server, and the page in your browser once it is ready. Keep the window
  open while you use it; to stop, press **Ctrl-C** in it (answer `Y` if it
  asks "Terminate batch job"). Closing the window stops it too, and any
  queued work with it.
- Offers to start AuraDream now. On the first visit press **Switch & load**
  in the Models panel ([step 7](#7-start-the-server-and-make-an-image)).

What it never does: download model weights, write `auradream.settings.json`,
move the gallery, set `AURADREAM_*` variables for your user, touch WSL, or
ask for administrator rights. The only user variables it may set are
`HF_HOME` and `UV_CACHE_DIR`, and only to a folder you chose.

To see what it would do without changing anything, run
`install.cmd -DryRun` from a terminal in the folder; `-Yes` takes every
default without asking. To update later, stop AuraDream and double-click
**`update.cmd`** ([Updating](#updating)).

## Before you start

Check the machine against this list first. Only one configuration has been
measured: an RTX 5080 with 16 GB of VRAM in a 64 GB Windows 11 machine.

- **GPU generation.** An NVIDIA card that torch's CUDA 13 (cu130) wheels are
  built for. The torch 2.13.0+cu130 the project installs lists compute
  capability 7.5 to 12.0 (`sm_75` to `sm_120`, from
  `torch.cuda.get_arch_list()`): the RTX 20 and GTX 16 series or newer.
  Older cards, such as the GTX 10 series (6.1), are not in that list, so
  expect them to fail at the first load even though `nvidia-smi` works.
  Only the RTX 5080 has been tested.
- **VRAM.** Only the 16 GB RTX 5080 is measured. The defaults use fp8 and
  model offload; **sequential** offload under **Models** uses less VRAM,
  more slowly. Whether a checkpoint fits an 8-12 GB card has not been
  measured; the SDXL anime checkpoints are the lightest.
- **RAM.** A cold load of the default, FLUX.1 Krea [dev] (fp8, model
  offload), peaked at **27.9 GiB working set and 35.4 GiB private**
  (memory committed, which RAM and the page file must cover) on the 64 GB
  test machine, and private rose to **39.9 GiB** near the end of the first
  1024x1024 render ([the fixed loader](verified.md#ram-at-the-load-the-fixed-loader-2026-10-27)).
  Other FLUX-size checkpoints (FLUX.2 klein 9B, Krea 2, Qwen-Image) and int8
  have not been measured with that loader, and no 32 GB machine has been
  measured. The SDXL anime checkpoints need far less.
- **Disk.** More than the weights:
  - the checkpoint itself in the Hugging Face cache: about 32 GiB for the
    default, several GB for an SDXL checkpoint;
  - a converted copy for some checkpoints, in the same cache under
    `auradream\converted`: Illustrious, WAI and RouWei each need about
    14 GB for the source and conversion together;
  - uv's package cache, `%LOCALAPPDATA%\uv\cache`: several GB, mostly torch
    (an estimate). With the project on another drive than the cache, uv
    copies the packages into `.venv` instead of linking them, so they take
    the space twice;
  - the project's `.venv`: about 3 GB (an estimate, from one checkout);
  - the optional prompt critic, on first use of **Ask the critic**: about
    4.5 GiB (4.9 GB), in the Hugging Face cache.

  For the default with the critic, plan on roughly 45-50 GB free (an
  estimate), plus room for your images.
- **Windows 11** (or WSL 2, above). A **Hugging Face account** only for the
  gated checkpoints, the default among them (step 6).

## 1. Check the GPU driver

Open PowerShell: press the Windows key, type `PowerShell` and press Enter
(or right-click the Start button and choose **Terminal**, which opens
PowerShell). Every command on this page is typed there. Run:

```powershell
nvidia-smi
```

The header's **CUDA Version** must be **13.0 or higher**. If it is lower,
or the command is not found, install the current driver from
[NVIDIA's driver page](https://www.nvidia.com/Download/index.aspx) and
reboot. You do not need the CUDA Toolkit: torch brings its own CUDA runtime.

## 2. Install Git and uv

[uv](https://docs.astral.sh/uv/) installs the right Python and every package.
With winget (built into Windows 11):

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
```

Or uv's own installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell so both are on your PATH, then check with
`git --version` and `uv --version`. You do not need to install Python
yourself: the project needs 3.13, and uv downloads it in step 4 if it is
missing.

Git is used only to get the code and to update it. GitHub's **Code >
Download ZIP** works too, but a ZIP has no simple way to update
([Updating](#updating)).

## 3. Choose where the downloads go

Checkpoint weights go to the Hugging Face cache, by default
`C:\Users\<you>\.cache\huggingface`, on drive C:. The converted copies, the
critic and your Hugging Face sign-in are kept there too. To keep them on
another drive, set `HF_HOME` for your user now, before anything downloads,
then close and reopen PowerShell:

```powershell
setx HF_HOME "D:\huggingface"
```

Skip this to keep the default. If you change it later, sign in again
(step 6), since the token is stored there.

## 4. Get the code and install

`git clone` makes an `AuraDream` folder inside the folder PowerShell is in,
shown before the `>`. A new PowerShell starts in `C:\Users\<you>`, so the
project lands in `C:\Users\<you>\AuraDream`; to put it somewhere else,
`cd` there first (for example `cd D:\`).

```powershell
git clone https://github.com/matt439/AuraDream.git
cd AuraDream
uv sync
```

`uv sync` creates `.venv` in the folder and installs torch from the CUDA 13
index, so it downloads a few GB. Check that torch sees the card:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and your GPU's name. `False` means the driver is too
old for CUDA 13 (step 1). A card older than the list above can still print
`True` and its name, with a warning that its CUDA capability "is not
compatible with the current PyTorch installation": that warning means the
card is too old ([Before you start](#before-you-start)).

## 5. Pick a first checkpoint

The two sensible first choices:

- **An SDXL anime checkpoint**, such as Animagine XL 4.0. It is open (no
  Hugging Face account needed), a much smaller download and far lighter on
  memory. Start here on a machine with less than the 64 GB measured, or
  just to check that everything works. The
  [anime model guide](anime-model-guide.md) compares the others.
- **FLUX.1 Krea [dev]**, the default: photographic, about 32 GiB of weights.
  Its load peaked at 27.9 GiB working set and 35.4 GiB private on the
  64 GB test machine, and private reached 39.9 GiB during the first render
  (the RAM item above). It is gated: it needs step 6.

Krea 2 Turbo and Raw are gated too, and need their authors' manual approval,
which can take a while.

## 6. Sign in to Hugging Face (gated checkpoints only)

Skip this for open checkpoints such as Animagine.

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Open the checkpoint's page, for the default
   [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev),
   and accept its licence. FLUX.1 Krea [dev] is a click-through and is
   non-commercial.
3. Create a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):
   **Create new token**, type **Read**.
4. From the project folder, run this and paste the token when asked:

   ```powershell
   uv run hf auth login
   ```

The token is saved in the Hugging Face cache for your Windows user (step 3),
so this is once per machine.

## 7. Start the server and make an image

For the default checkpoint:

```powershell
uv run python -m backend.main
```

For an open checkpoint, pass its id:

```powershell
uv run python -m backend.main --model cagliostrolab/animagine-xl-4.0
```

Open **<http://127.0.0.1:7860>**. On a first visit the **Models** panel
opens by itself with the checkpoint you started with selected. Press
**Switch & load**: it loads that checkpoint and saves the choice, so the
next plain start comes back to it. Closing the panel instead does not save
the choice, and the next start without `--model` returns to the default.

The first load downloads the weights: 32 GiB takes about 45 minutes at
100 Mbit/s, so scale that for your connection. Some checkpoints then convert
on the CPU, which adds minutes more. The terminal shows the progress. Later
loads come from the cache; the default's took about 70 s on the test
machine.

Then enter a prompt, choose a size and press **Generate**. **Examples** and
the **?** hints show the checkpoint's own prompting style and sizes; Steps,
CFG and Mu start at its defaults. Images and the gallery database are saved
in the project folder (`outputs/` and `auradream.db`).

Stop the server with Ctrl-C. To start again, open PowerShell, go to the
project folder and run the server:

```powershell
cd AuraDream
uv run python -m backend.main
```

It restores the checkpoint and settings you last saved in **Models**.

## If something goes wrong

First, from the project folder, run the read-only machine check:

```powershell
uv run python -m backend.doctor
```

It prints a PASS, WARN or FAIL line for the GPU and PyTorch, the driver's
CUDA version, RAM, free disk where the downloads and images go, the port,
and whether a Hugging Face sign-in is present, with what to do under
anything that is not a pass. It downloads and loads nothing and never
contacts Hugging Face.

- **"is gated on Hugging Face and this sign-in has no access yet"**: accept
  the licence (or wait for approval) on the checkpoint's page, run
  `uv run hf auth login` with a read token, then load again.
- **`torch.cuda.is_available()` is `False`, or a CUDA error at load**:
  update the NVIDIA driver (step 1), and check the card's generation
  ([Before you start](#before-you-start)). Do not install a separate CUDA
  Toolkit to fix it.
- **The machine slows to a crawl or the load fails partway on a FLUX-size
  checkpoint**: it is out of RAM. Close other large programs and try again,
  or start with an SDXL checkpoint (step 5).
- **Out of GPU memory while generating**: use a smaller size or batch, or
  set offload to **sequential** under **Models** (slower, less VRAM).
- **"uv: command not found" / not recognised**: open a new PowerShell after
  installing uv (step 2).
- **The page does not open**: the server prints its address when it is
  ready; use <http://127.0.0.1:7860> in a browser on the same machine.

## Updating

Your images, gallery and settings are not part of the code, so an update
leaves them where they are. Stop the server first, then double-click
**`update.cmd`** in the project folder. It refuses while AuraDream is
running, copies the gallery database as below, runs `git pull --ff-only`
and `uv sync`, and explains a pull Git refuses without discarding anything
(`update.cmd -DryRun` shows what it would do). By hand, from the project
folder:

```powershell
Copy-Item auradream.db auradream-before-update.db
git pull --ff-only
uv run python -m backend.main
```

The copy matters: a new version can add to the gallery database on its
first start, and an older version cannot undo that, so the copy is the way
back. (A gallery from before the rename may be `local-diffusion.db`; copy
that one instead, to `local-diffusion-before-update.db`.) Git ignores
both copies. The next start installs any changed packages before it
runs, which can take a while when torch changes. If `git pull --ff-only`
refuses, a file of the code was changed in your folder; `git status` names
it.

A ZIP download cannot be updated in place. Running `install.cmd` again
offers to turn the folder into a Git checkout ([Easy install](#easy-install)),
after which `update.cmd` works. Otherwise, download the new ZIP, extract it
to a new folder, run `uv sync` there, and copy across from the old folder:
`auradream.db`, `outputs\`, `loras\`, `bundles\`,
`auradream.settings.json` and the other `auradream.*.json` files (saved
setups, snippets, saved searches and backup settings).

## Uninstall

1. Keep what you want first: copy `outputs\` and `auradream.db`, or save a
   bundle (**Bundles…** under the gallery's search box).
2. Delete the project folder. The gallery, settings, LoRAs and `.venv` are
   inside it, unless an `AURADREAM_*` path variable moved them.
3. Delete the Hugging Face cache: `C:\Users\<you>\.cache\huggingface`, or
   the folder `HF_HOME` names. It holds the weights (under `hub\`), the
   converted copies and the critic's llama.cpp (under `auradream\`) and
   your token. If other programs use that cache, delete only `auradream\`
   (or an older `local-diffusion\`) and the checkpoints' and critic's
   `models--…` folders under `hub\`. If you set `HF_HOME`, remove it with
   `[Environment]::SetEnvironmentVariable("HF_HOME", $null, "User")`.
   The token itself can be deleted at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Empty uv's cache with `uv cache clean`, which also removes other
   projects' cached packages (`uv cache prune` removes only unused
   entries). `uv python uninstall 3.13` removes the Python uv installed.
5. Delete the **AuraDream** shortcuts the installer made: on the desktop,
   and in the Start menu's folder
   (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\AuraDream.lnk`). If
   the installer set `UV_CACHE_DIR`, remove it as `HF_HOME` above and
   delete that folder.
6. If you added `bin\` to your user PATH, remove that entry. Uninstall uv
   and Git (`winget uninstall astral-sh.uv`, `winget uninstall Git.Git`) if
   nothing else needs them.

## Next

- [Usage guide](usage.md): prompts, batches, grids, refine, inpaint and
  outpaint, the gallery and the CLI.
- [Settings](../README.md#settings): environment variables, including
  quantization, offload, and the output and database paths.
- The installer's shortcuts start AuraDream without a terminal. To start it
  from a terminal in any folder as well (optional), add the repository's
  `bin\` folder to your user PATH and run `auradream --open`
  (`auradream-wsl` for WSL 2) ([details](../README.md#quick-start)).

## If you used it before the rename

The project was called Local Diffusion, in a repository named
ImageGeneratorLocal, until 2026-10-27. A clone made before then needs only
its remote pointed at the new address
(`git remote set-url origin https://github.com/matt439/AuraDream.git`); the
folder can keep its old name. A folder moved or renamed to match needs two
things done once: point the user PATH entry at the new `bin\`, and recreate
the virtual environment (delete `.venv`, then `uv sync`), whose console
scripts still name the old folder's Python. The default gallery, LoRAs
and settings move with the folder; an `AURADREAM_*` path variable that
names the old folder needs the new one. The `bin\` commands were
`local-diffusion` and `local-diffusion-wsl` before the rename; a PATH entry
for an existing clone's `bin\` needs no change unless the folder itself
moved.
