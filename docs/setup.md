# First-time setup

A step-by-step path from a fresh Windows 11 machine to a first image. The
[README](../README.md) has every launch option; this page covers only what a
first run needs, in order. Allow some time for the first image: a checkpoint
download is several GB, and the default is about 32 GiB.

For the 4-bit Nunchaku route under WSL 2, see
[Run under WSL 2](../README.md#run-under-wsl-2-4-bit-nunchaku) instead;
`wsl/setup.sh` does the driver check, uv and the install for you there. For
a Linux PC that serves the page to others on the network, see
[Native Linux](#native-linux-a-server-on-your-network) near the end of this
page; its scripts do the same there.

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
   already done and keeps the choices saved in the Models panel. It refuses
   while AuraDream is running (stop it first) and on a network share (use
   a folder on this machine's own drive).

What it does, in order:

- Checks Windows, the folder (OneDrive, a very long path or non-English
  characters get a warning), the NVIDIA driver (CUDA 13.0 or higher, from
  `nvidia-smi`), the card's generation (compute capability 7.5 or newer)
  and video memory, and the RAM, which it compares with the measured
  figures below; it promises nothing for other machines. A driver that is
  too old or a card that is too old stops it with what to do.
- Installs **uv** and **Git** only if they are missing, with winget, or
  with uv's own installer (a pinned version) where winget is not available.
  Git is installed for your user only; where winget cannot do that, it says
  so and AuraDream runs without Git (only `update.cmd` needs it).
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

Check the machine against this list first. The configuration measured
throughout is an RTX 5080 with 16 GB of VRAM in a 64 GB Windows 11 machine;
an 8 GB RTX 3070 on native Ubuntu has one SDXL checkpoint measured
([An 8 GB card](#an-8-gb-card)).

- **GPU generation.** An NVIDIA card that torch's CUDA 13 (cu130) wheels are
  built for. The torch 2.13.0+cu130 the project installs lists compute
  capability 7.5 to 12.0 (`sm_75` to `sm_120`, from
  `torch.cuda.get_arch_list()`): the RTX 20 and GTX 16 series or newer.
  Older cards, such as the GTX 10 series (6.1), are not in that list, so
  expect them to fail at the first load even though `nvidia-smi` works.
  The RTX 5080 (12.0) and an RTX 3070 (8.6) have been tested.
- **VRAM.** The 16 GB RTX 5080 is measured throughout. The defaults use fp8
  and model offload; **sequential** offload under **Models** uses less VRAM,
  more slowly. On an 8 GB RTX 3070, Animagine XL 4.0 at fp8 with model
  offload peaked at 4.2 GiB of the whole card (5.3 GiB read mid-job for a
  batch of four); no other checkpoint has been measured on 8-12 GB, and the
  FLUX-size ones are not expected to fit 8 GB at fp8 with model offload (not
  measured; sequential offload untried; [An 8 GB card](#an-8-gb-card)). The SDXL anime checkpoints are the
  lightest.
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
- **Windows 11** (or WSL 2, or [native Linux](#native-linux-a-server-on-your-network)).
  A **Hugging Face account** only for the gated checkpoints, the default
  among them (step 6).

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
which can take a while. FLUX 3 [Klein] and Klein Base are click-throughs
under the FLUX Kommunity License; how much memory they need has not been
measured.

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

If you started without `--model` and have not signed in to Hugging Face
(step 6), the panel selects the open **Z-Image Turbo** instead of the
default, which needs a sign-in, and its note says so. Nothing is saved
until you press **Switch & load** or **Apply & save**; **Revert** puts the
default back, and any other row can be chosen.

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

## Native Linux (a server on your network)

A Linux PC with an NVIDIA card can run AuraDream as a service that other
PCs on the network open in a browser, with nobody logged in at it. Three
scripts in `linux/` do the work; each is safe to run again, never uses
`sudo` (a command that needs it is printed for you to run) and takes
`--dry-run` to print what it would do and change nothing. Measured on
Ubuntu 26.04 with an 8 GB RTX 3070
([what was measured](verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03));
other distributions with systemd should work the same, untested.

### What it needs

- **The distribution's NVIDIA driver**, whose `nvidia-smi` reports **CUDA
  13.0 or higher**. On Ubuntu, `sudo apt install nvidia-driver-595-open`,
  or on a machine without a desktop
  `sudo apt install nvidia-headless-595-open nvidia-utils-595`, then
  reboot. Do not install the CUDA Toolkit: torch brings its own CUDA
  runtime. The card's generation is as in
  [Before you start](#before-you-start) (compute capability 7.5 or newer).
- **Git** and **curl** (`sudo apt install git curl`). uv and Python 3.13
  are installed by the setup script.
- **gcc**, usually present (`sudo apt install gcc` if not): torch 2.13
  compiles a small C helper for its Triton kernels the first time some
  checkpoints render.
- For the optional prompt critic: glibc 2.38 or newer (Ubuntu 24.04 or
  newer) and `libgomp1` and `libssl3`; the setup script says if any is
  missing.
- **Node is not needed** to run AuraDream; only the test suite's page
  checks use it.
- RAM and disk as in [Before you start](#before-you-start).

### Install

Run every command as your own user, from a login of your own (the console
or `ssh`), not under `su` or `sudo`:

```bash
git clone https://github.com/matt439/AuraDream.git
cd AuraDream
bash linux/setup.sh
```

`setup.sh` refuses under WSL (that is `wsl/setup.sh`), as root, or while
AuraDream runs. It checks the driver's CUDA version, the card's generation
and memory, a C compiler and the critic's libraries; installs uv only if it
is missing (uv's own installer, the version `install.ps1` pins); runs
`uv python install 3.13` and `uv sync` (a few GB of packages, never model
weights); checks that torch sees the card; and runs the read-only
`python -m backend.doctor`. It never writes `auradream.settings.json` or
moves the gallery. If it installed uv, run the line it prints
(`. "$HOME/.local/bin/env"`) or open a new shell.

To try it on the machine itself first: `uv run python -m backend.main`, then
<http://127.0.0.1:7860> in a browser there. Sign in to Hugging Face
([step 6](#6-sign-in-to-hugging-face-gated-checkpoints-only),
`uv run hf auth login`) only for the gated checkpoints.

### Run it as a service

```bash
bash linux/install-service.sh
sudo loginctl enable-linger $USER
```

`install-service.sh` writes the systemd user unit
`~/.config/systemd/user/auradream.service`, enables it and starts it
(`--no-start` only enables it). The unit runs `uv run --no-sync python -m
backend.main` in the checkout, restarts it after a failure, and gives a stop
two minutes to finish the current step and any gallery bundle operation.
**Lingering** (the `loginctl` line) makes it start at boot and keep running
with nobody logged in; without it, the service runs only while you are
logged in. The script says which you have.

The service's settings live in `~/.config/auradream/auradream.env`, readable
by you only. The script writes it once, when there is none, and never
rewrites it: edit it, then `systemctl --user restart auradream`. One
`NAME=value` per line, no quotes, absolute paths only (systemd expands
nothing there). It holds:

- `AURADREAM_HOST=0.0.0.0`, so other PCs can reach the port.
- `AURADREAM_ALLOWED_HOSTS`, the names the page may be opened by: loopback,
  this machine's name, its `.local` name and its IPv4 addresses when the
  script ran, for example
  `127.0.0.1,localhost,my-box,my-box.local,192.168.1.20`. Add a name or
  address another PC uses (after a DHCP change, say). Keep `127.0.0.1` (and
  `localhost`): a second start, the doctor and the command line ask the
  server at `127.0.0.1`, and without it they take AuraDream for another
  program.
  Write every name exactly and in lowercase, as browsers send it; a
  wildcard (`*` or `*.local`) loads the page and then refuses every change
  it asks for. The doctor warns about a missing `127.0.0.1`, capitals and
  wildcards.
- The xet downloader's fixed connection count (as `wsl/env.sh` sets it),
  and commented examples for `HF_HOME`, `AURADREAM_PORT` and the two
  keep-awake settings.

At the end the script prints the addresses to open from another PC, such as
`http://my-box:7860/` or `http://192.168.1.20:7860/`. A `.local` name needs
mDNS (`avahi-daemon`); if a firewall is on, allow the port
(`sudo ufw allow 7860/tcp`). **There is no login**: anyone who can reach the
port can read and delete the gallery and make the machine download
checkpoints with its Hugging Face token. Only do this on a network you
trust.

```bash
systemctl --user status auradream          # its state
journalctl --user -u auradream -f          # follow its log
systemctl --user stop auradream            # stop / start / restart
```

A page opened by a LAN address over plain `http` is not a secure context
for the browser, so the context menu's copy items and desktop notifications
are held, each with its reason; everything else works. Unencrypted
automatic backups use the ordinary bundle path on Linux (not run there
yet); encrypted automatic
backups (the Windows credential manager) and backup cleanup (an NTFS shared
mode) are Windows only.

### Keeping the PC awake

A PC that suspends after a spell of idleness sees no keyboard or mouse while
a browser elsewhere drives it. So while a job, a load, a switch, a gallery
bundle or the critic's download runs, and for 20 minutes after the last
request someone made, the server holds a `systemd-inhibit` block on sleep
and idle; the page's own timed polls do not count, so an idle open tab lets
the PC sleep. `AURADREAM_KEEP_AWAKE_IDLE_MINUTES` changes the 20 (`0` holds
only while busy); `AURADREAM_KEEP_AWAKE=false` turns it off.
`systemd-inhibit --list` shows the hold as `AuraDream`. Under a service,
polkit may refuse it; the doctor's **Keep awake** row says so and prints a
rule that allows it. Waking the PC from the network is your own setup
(Wake-on-LAN, or a wake gate of your own), outside the server;
[auradream-server](#from-another-pc-auradream-server) can send the wake.

### Updating

```bash
bash linux/update.sh --restart
```

`--restart` stops the service only when it is idle (no job, load, switch or
gallery bundle operation), updates, and starts it again even if the update
fails; without it, `update.sh` refuses while AuraDream runs. It copies the
gallery database to its before-update name (the way back, as in
[Updating](#updating)), runs `git pull --ff-only` and `uv sync`, and never
resets, stashes or checks out a change of yours: a pull Git refuses is
explained. It reads the service's environment file, so it copies the
database the service uses. `install-service.sh` again rewrites the unit
(never the environment file).

### Controlling the service

On the server, from the checkout, `linux/control.sh` does the everyday
work:

```bash
bash linux/control.sh status            # the service, the checkout, the server's own status
bash linux/control.sh restart           # refused while it is busy; --force interrupts the work
bash linux/control.sh logs -n 50 -f     # the last 50 lines of its log, then follow it
bash linux/control.sh doctor            # the read-only machine check, with the service's settings
```

`status` shows the service's state, since when, whether it is enabled and
whether lingering is on; the checkout's branch and commit and how far it is
behind GitHub (after a quiet `git fetch`; a failed fetch is only noted);
and the running server's own answer on `127.0.0.1`: the checkpoint and how
it loads, the model's state, active jobs, the card's holder, keep awake and
why, the card's memory and the free disk, then AuraDream's line from
`systemd-inhibit --list`. It exits 0 when the service is active and
answering, 1 otherwise. `stop` and `restart` ask the server first and
refuse while a job, a load, a switch, something holding the card or a
gallery bundle operation is under way (as `update.sh --restart` does);
`--force` stops it anyway and says what it interrupts. `start` and
`restart` wait up to a minute for the page to answer. `logs` takes
`-n N` (default 100) and `-f`.

### From another PC: auradream-server

`bin\auradream-server` runs those commands (and `update.sh`) on the server
over `ssh` from a Windows PC: with the repository's `bin\` on the PATH,
`auradream-server status` works from any folder. It needs Windows' OpenSSH
client (built in) and a key the server accepts, since it never waits for a
password (`ssh-keygen`, then add the `.pub` file's line to the server's
`~/.ssh/authorized_keys`). Its settings are environment variables, read by
this script only:

- `AURADREAM_SERVER` (required): the server as `user@host` or `host`, for
  example `me@my-box` or `me@192.168.1.20`; a `Host` alias from
  `~/.ssh/config` works too.
- `AURADREAM_SERVER_DIR`: the checkout on the server (default `~/AuraDream`).
- `AURADREAM_SERVER_PORT`: the server's port (default 7860), for `open`.
- `AURADREAM_SERVER_WAKE` (optional): how to wake a sleeping server, either
  a URL that is POSTed (a wake gate of your own on the network, for example
  `http://192.168.1.5:8650/__wake`) or the server's MAC address
  (`aa:bb:cc:dd:ee:ff` or `aa-bb-cc-dd-ee-ff`), for which a Wake-on-LAN
  packet is broadcast on this PC's network (port 9). Wake-on-LAN must be on
  in the server's firmware and network settings.

```powershell
$env:AURADREAM_SERVER = 'me@my-box'                        # this window; setx for every new one
$env:AURADREAM_SERVER_WAKE = 'aa:bb:cc:dd:ee:ff'           # optional
auradream-server status
auradream-server restart            # -Force: even while it is busy
auradream-server stop               # start, stop, restart
auradream-server update             # bash linux/update.sh --restart; -DryRun: --restart --dry-run
auradream-server logs -Lines 50     # -Follow keeps reading; Ctrl-C ends it
auradream-server doctor
auradream-server connect            # a shell in the checkout
auradream-server connect -Command 'git log -3 --oneline'
auradream-server open               # http://my-box:7860/ in this PC's browser
```

Before each command it checks that the server's ssh port answers; if not,
it wakes the server (when `AURADREAM_SERVER_WAKE` is set) and waits up to
three minutes, waking again every 15 seconds, then lets `ssh` try. It exits
with the server's exit code (255 when `ssh` itself fails, with a hint).
Arguments with double quotes are refused, since Windows PowerShell 5.1 does
not pass them to `ssh` intact; use single quotes inside `-Command`. A
checkout from before `control.sh` existed says so: `auradream-server update`
brings it.

### Downloads on another disk

Checkpoints go to `~/.cache/huggingface`. To keep them on another disk, set
`HF_HOME` to an absolute path there in the environment file (and `export`
it in a terminal you run `uv` from), then restart the service. Move an
existing `~/.cache/huggingface` there first to keep what is downloaded, and
your Hugging Face sign-in with it.

### An 8 GB card

On an RTX 3070 (8 GB), Animagine XL 4.0 at fp8 with model offload made two
1024x1024 images in 30 s and peaked at 4.2 GiB (4,289 MiB) of the whole
card; a batch of four through the service read about 5.3 GiB (5,421 MiB)
mid-job. Start with the SDXL anime
checkpoints, Anima or FLUX.2 [klein] 4B (only the SDXL one is measured
there). fp8 works on cards before the RTX 40 series: torchao's fp8
weight-only quantization keeps the weights in fp8 and computes in bf16, so
it needs no fp8 tensor cores. The FLUX-size checkpoints (FLUX.1, Krea 2,
klein 9B, Qwen-Image) are not expected to fit 8 GB at fp8 with model
offload (not measured), since model offload puts a whole transformer on the
card: the default peaked at 12.0 GiB on the 5080.
Sequential offload has not been tried on 8 GB, and the page gives these no
fit warning yet. Nunchaku 4-bit (`AURADREAM_QUANT=nunchaku`)
needs the `int4` repos below an RTX 50 card (the doctor names them); the
`nvfp4` ones need compute capability 10.0 or higher. Nunchaku on an RTX 30
card has not been measured.

### Uninstalling the service

```bash
bash linux/install-service.sh --uninstall
```

stops, disables and removes the unit. It keeps the environment file (delete
`~/.config/auradream/` yourself) and leaves the gallery, the checkpoints and
the checkout alone; [Uninstall](#uninstall) covers those, with the
checkout folder and `~/.cache/huggingface` (or `HF_HOME`) in place of the
Windows folders and no shortcuts or user variables to remove. `sudo loginctl disable-linger
$USER` turns lingering off if nothing else needs it.

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
