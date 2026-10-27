# First-time setup

A step-by-step path from a fresh Windows 11 machine to a first image. The
[README](../README.md) has the full requirements and every launch option;
this page covers only what a first run needs, in order. Allow some time
for the first image: a checkpoint download is several GB, and the default is
about 32 GiB.

For the 4-bit Nunchaku route under WSL 2, see
[Run under WSL 2](../README.md#run-under-wsl-2-4-bit-nunchaku) instead;
`wsl/setup.sh` does steps 1 to 3 for you there.

## 1. Check the GPU driver

Open PowerShell and run:

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
yourself: the project needs 3.13, and uv downloads it in the next step if it
is missing.

## 3. Get the code and install

```powershell
git clone https://github.com/matt439/AuraDream.git
cd AuraDream
uv sync
```

The project was called Local Diffusion, in a repository named
ImageGeneratorLocal, until 2026-10-27. A clone made before then needs only
its remote pointed at the new address
(`git remote set-url origin https://github.com/matt439/AuraDream.git`); the
folder can keep its old name.

`uv sync` creates `.venv` in the folder and installs torch from the CUDA 13
index, so it downloads a few GB. Check that torch sees the card:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and your GPU's name. `False` means the driver is too
old for CUDA 13 (step 1).

## 4. Pick a first checkpoint

The two sensible first choices:

- **An SDXL anime checkpoint**, such as Animagine XL 4.0. It is open (no
  Hugging Face account needed), a much smaller download and far lighter on
  memory. Start here if the machine has less than 64 GB of RAM, or just to
  check that everything works. The
  [anime model guide](anime-model-guide.md) compares the others.
- **FLUX.1 Krea [dev]**, the default: photographic, about 32 GiB of weights.
  A cold load of it has peaked around 46 GiB of RAM on the 64 GB test
  machine ([RAM at the load](verified.md#ram-at-the-load-2026-10-25)), so
  plan on **64 GB or more**. It is gated: it needs step 5.

Krea 2 Turbo and Raw are gated too, and need their authors' manual approval,
which can take a while.

## 5. Sign in to Hugging Face (gated checkpoints only)

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

The token is saved for your Windows user, so this is once per machine.

## 6. Start the server and make an image

For the default checkpoint:

```powershell
uv run python -m backend.main
```

For an open checkpoint, pass its id:

```powershell
uv run python -m backend.main --model cagliostrolab/animagine-xl-4.0
```

Open **<http://127.0.0.1:7860>**. On a first visit the **Models** panel
opens by itself; the checkpoint you started with is selected, so press
**Switch & load**, or close the panel and press **Generate**. Either way the
first load downloads the weights and, for some checkpoints, converts them
on the CPU, which can take several minutes. The terminal shows the
progress. Later loads come from the cache and take far less.

Then enter a prompt, choose a size and press **Generate**. **Examples** and
the **?** hints show the checkpoint's own prompting style and sizes; Steps,
CFG and Mu start at its defaults. Images and the gallery database are saved
in the project folder (`outputs/` and `auradream.db`).

Stop the server with Ctrl-C. Next time, `uv run python -m backend.main`
alone restores the checkpoint and settings you last saved in **Models**.

## Where the downloads go

Checkpoint weights go to the Hugging Face cache, by default
`C:\Users\<you>\.cache\huggingface`, on drive C:. To keep them on another
drive, set `HF_HOME` for your user before the first download, then open a
new PowerShell:

```powershell
setx HF_HOME "D:\huggingface"
```

Sign in again (step 5) after changing it, since the token is stored there
too. Checkpoints that convert on first load (Illustrious, WAI, RouWei and
others) keep the converted copy in the same cache, under
`auradream\converted` (or the `local-diffusion\converted` an earlier
install made, which stays in use); Illustrious, WAI and RouWei each need about
14 GB for the source and conversion together.

## If something goes wrong

- **"is gated on Hugging Face and this sign-in has no access yet"**: accept
  the licence (or wait for approval) on the checkpoint's page, run
  `uv run hf auth login` with a read token, then load again.
- **`torch.cuda.is_available()` is `False`, or a CUDA error at load**:
  update the NVIDIA driver (step 1). Do not install a separate CUDA Toolkit
  to fix it.
- **The machine slows to a crawl or the load fails partway on a FLUX-size
  checkpoint**: it is out of RAM. Close other large programs and try again,
  or start with an SDXL checkpoint (step 4).
- **Out of GPU memory while generating**: use a smaller size or batch, or
  set offload to **sequential** under **Models** (slower, less VRAM).
- **"uv: command not found" / not recognised**: open a new PowerShell after
  installing uv (step 2).
- **The page does not open**: the server prints its address when it is
  ready; use <http://127.0.0.1:7860> in a browser on the same machine.

## Next

- [Usage guide](usage.md): prompts, batches, grids, refine, inpaint and
  outpaint, the gallery and the CLI.
- [Settings](../README.md#settings): environment variables, including
  quantization, offload, and the output and database paths.
- To start from any folder, add the repository's `bin\` folder to your
  user PATH and run `auradream` (`auradream-wsl` for WSL 2; they were
  `local-diffusion` and `local-diffusion-wsl` before the rename, and a
  PATH entry for an existing clone's `bin\` needs no change)
  ([details](../README.md#quick-start)).
