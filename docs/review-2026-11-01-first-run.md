# Review 2026-11-01: first-run experience and an installer

Matt asked what AuraDream is like for a first-time user who is not confident
with git, building or terminal commands, and whether an automated installer
would help. Seven read-only agents walked the setup (a novice following
`docs/setup.md` literally, failure paths, day-2 use), surveyed how comparable
tools install, wrote two independent proposals and had them checked by a
skeptical reviewer. Nothing was changed. Line numbers are as of b6c0fe0.

Answer: an installer is worth building, but third. The worst first-run
problems are wrong docs and a few plain-words gaps in the backend, which are
cheaper and which the installer would otherwise have to paper over.

## What a first-timer meets today

From the GitHub page to a first image is about 19 actions with an open
checkpoint (Animagine) and about 24 with the gated default, FLUX.1 Krea [dev]
(an estimate from the walkthrough, not a measurement). It takes PowerShell,
`nvidia-smi` and the driver, winget, Git, uv, the `hf` CLI and a browser, plus
a Hugging Face account and token for the default. Everything but the browser
part is typed.

Good already: `docs/setup.md` checks each step's result (`git`/`uv --version`,
the torch CUDA check), says not to install the CUDA Toolkit or Python, steers
low-RAM users to an SDXL checkpoint, and its troubleshooting quotes match the
code's wording (`setup.md:162` and `backend/hub_access.py:138`; the gated
messages at `backend/main.py:166-176`).

## Findings

### F1 RAM advice is out of date, and names a guard that does not exist (high)

`README.md:20-24` says a cold FLUX-class load "peaked around 46 GiB ... where
it stopped at the app's free-RAM guard" and to plan on 64 GB;
`docs/setup.md:84-87` repeats the figures. `docs/verified.md:717` says that
floor was the harnesses' 4 GiB free-RAM floor (`tools/card_guard.py`), and
`verified.md:770` gives the fixed loader's peak: 27.9 GiB working set, 35.4
GiB private. `backend/` has no RAM check (no `virtual_memory`/psutil use;
`model_info.py:181` says "a free-RAM guard", ambiguous rather than false).
The fix states the measured numbers and that a 32 GB machine is unmeasured;
it must not become a new promise.

### F2 Closing the first-visit Models panel does not save the choice (high)

`docs/setup.md:127-129` offers "close the panel and press Generate" beside
Switch & load. Only the switch route saves (`startup_preferences.save` at
`backend/main.py:1020`, reached from `web/model-switch.js:1140`), and
`setup.md:139-140` promises the last saved choice returns. A user who started
on an open checkpoint that way gets the gated default on the next plain start.

### F3 No preflight on Windows; failures arrive after Generate (high)

There is no Windows counterpart to `wsl/setup.sh:16-18,48` (which checks that
`nvidia-smi` runs and `torch.cuda`, not the CUDA version), no doctor command,
no RAM or disk precheck. The server starts and the first problem appears at
the first load, after a wait:

- The CUDA refusal suggests CPU settings and never mentions the driver
  (`backend/backends/diffusers_backend.py:505-509`); a rewrite keeps the
  `AURADREAM_DEVICE`/offload explanation (comment at :498-501).
- An invalid or expired token is a raw error on the load path:
  `_HUB_FAILURES` (`backend/main.py:114-116`) maps Gated, NotFound and the two
  offline errors only; `hub_access`'s denied wording (`hub_access.py:109-112,
  140-142`) runs only on a switch to a different checkpoint (`main.py:979`).
- A busy port, including a second launch of a running server, prints a raw
  uvicorn bind error (`main.py:1171-1176` catches only `KeyboardInterrupt`).
- The status's free-disk figure measures `outputs_dir` (`main.py:720-727`),
  not the model cache's drive.

### F4 Daily use needs a terminal (high)

No desktop or Start-menu shortcut; nothing opens the browser (no `webbrowser`
in `backend/`). Every start is `cd` + `uv run ...`, or `auradream` once
`bin\` is on PATH, which the docs never explain how to do (`README.md:95-100`,
`setup.md:184-189`). A shortcut to `bin\auradream.cmd` would avoid the
execution policy entirely (`bin/auradream.cmd:5-6`); the `.ps1` is reached
only from PATH in PowerShell.

### F5 Updating is undefined (high)

No tags, releases or changelog; `pyproject.toml:3` has stayed at 0.1.0 while
main moved 402 commits since 2026-10-25. `git pull` mostly works, since user
data is gitignored and `uv run` re-syncs, but nothing says so. A ZIP download
has no update path short of copying `auradream.db`, `outputs/`, `loras/`,
`auradream.settings.json` and the `auradream.*.json` sidecars by hand. A pull
can bring a `storage._migrate` step the user cannot undo.

### F6 Smaller doc gaps (medium)

- How to open PowerShell, where `git clone` puts the folder, and the missing
  `cd AuraDream` in the "start again" step (`setup.md:15, 50-51, 139-140`).
- Git is needed only for the clone; a ZIP would do (but see F5).
- The rename paragraph (`setup.md:55-64, 186-188`) sits mid-install.
- "Where the downloads go" (`HF_HOME`, `setup.md:142-158`) comes after the
  step that downloads.
- "Several minutes" for the first load (`setup.md:131`): 32 GiB is about 45
  minutes at 100 Mbit/s.
- No Updating or Uninstall section.
- The real disk total is the model cache, converted copies (about 14 GB for
  some checkpoints, under `HF_HOME\auradream\converted`), uv's cache
  (`%LOCALAPPDATA%\uv\cache`, several GB of torch wheels; duplicated when the
  repo is on another drive, since uv cannot hardlink across drives), `.venv`
  and the critic (about 4.9 GB); the docs give only the 32 GiB weights.

### F7 Hardware the docs do not address (medium)

- Card generation: CUDA 13 wheels are believed to drop GPUs below compute
  capability 7.5 (GTX 10 series and older; not verified this session). Such a
  card passes the `nvidia-smi` check and fails later.
- 8-12 GB cards, the common newcomer case, have no guidance; only the 16 GB
  RTX 5080 is measured, and fit on smaller cards is the card's to say.
- The gated default (`backend/config.py:76-78`) is the first thing a user
  without a token meets.

### F8 Not covered elsewhere (low)

Defender real-time scanning and corporate proxies/TLS interception slow or
break uv and Hub downloads; closing the console window mid-job kills queued
work, and under the `.cmd` Ctrl-C asks "Terminate batch job (Y/N)?".

## Prior art (2026)

ComfyUI ships a portable 7z and a Desktop app; Stability Matrix and Pinokio
are third-party launchers; A1111/Forge use `webui-user.bat`; InvokeAI a
launcher; SwarmUI a plain `install-windows.bat` in the repo. The low-maintenance
pattern for a solo developer is SwarmUI's: a plain-text script in the repo
that orders existing tools. Unsigned `.exe` installers meet SmartScreen; a
signing certificate costs about $150-300 a year. A downloaded `.cmd` or ZIP
still carries Mark-of-the-Web, so a reputation prompt is possible either way.

## Stages

Each stage is self-contained; do them in order.

1. **Docs (`DEV#27`, S).** F1's measured RAM wording; F2's route (recommend
   Switch & load as the finish); F6's gaps; a requirements box covering card
   generation (RTX 20 series or newer, once verified), VRAM, RAM and the real
   disk total including uv's cache; Updating and Uninstall sections. Docs only.
2. **First-visit panel (`UI#39`, S).** The fresh-visit Models panel recommends
   Switch & load and stops presenting closing as an equal finish, so the
   choice is saved. Smoke: `smoke-model-switch`/`smoke-model-startup`.
   Built 2026-11-01: the automatic opening's note leads with "Switch & load
   saves this choice for the next start; closing the panel saves nothing."
   until a press is accepted or the picker closes (`modelPickerOffered` in
   `web/model-switch.js`; Switch & load was already the primary button). No
   layout change; `smoke-model-startup` asserts it and screenshots it at
   1400x900. The preselected checkpoint is unchanged (`GEN#41`).
3. **Plain-words failures (`GEN#40`, M).** The CUDA refusal names the driver
   (and the card's compute capability where torch can read it) and keeps the
   offload text; 401/403 on the load path reuse `hub_access`'s wording; a
   busy port is detected, and when AuraDream is already the listener the
   second launch opens the browser and exits cleanly; an opt-in `--open` that
   opens the browser from a daemon thread after uvicorn has bound. Unit tests
   with fakes; no Hub calls.
4. **Installer (`DEV#28`, L).** `install.cmd` (runs `install.ps1` with
   `-ExecutionPolicy Bypass`), `install.ps1` and `update.cmd`, plain text in
   the repo.
   - Checks: `nvidia-smi` runs (a CUDA-version parse failure warns, never
     fails), card generation, RAM; winget may be missing on Windows 10, LTSC
     or managed images, so fall back to uv's version-pinned installer.
   - Installs uv and Git only if missing; a short install folder outside
     OneDrive, on the same drive as uv's cache or with `UV_CACHE_DIR` set.
   - Asks for `HF_HOME` before any download; `setx` reaches only later
     shells, so the script also sets `$env:HF_HOME` itself.
   - `uv sync`, then the `torch.cuda` check.
   - Guides the Hugging Face steps (opens the account, licence and token
     pages, then `hf auth login`); accepting a licence and creating a token
     are the user's own acts and cannot be automated.
   - Start-menu and desktop shortcuts to `bin\auradream.cmd --open`, titled
     with how to stop safely; a first launch on an open checkpoint.
   - `update.cmd` copies `auradream.db` before `git pull --ff-only`.
   - Never downloads weights, writes `auradream.settings.json`, moves the
     gallery, sets `AURADREAM_*` user-wide, or touches WSL.
   - Only the checks it needs go in a small `backend/doctor.py`; free space
     reads `HF_HUB_CACHE` and the converted copies.
   - Windows Sandbox has no GPU: test as a dry run with a stub `nvidia-smi`;
     the CUDA path is never exercised in CI.
5. **Later (`DEV#29`, deferred).** Hand-cut version tags so `update.cmd` can
   follow a stable point; a Pinokio script.

Decision for Matt (`GEN#41`): should a fresh browser visit with no stored
Hugging Face token preselect an open checkpoint instead of the gated default?

## Not taken

Recorded so they do not come back as rows: a browser wizard that accepts the
Hugging Face token (a new secret-handling route on a server with no login;
`huggingface_hub.login` writes into `HF_HOME`, so moving it later loses the
sign-in; no room in the panel at 1400x900); a release ZIP with a bundled
`uv.exe` and an update check (release discipline the current pace does not
have, a third-party binary, a new network call); `setx`-relocating
`AURADREAM_OUTPUTS_DIR`/`AURADREAM_DB_PATH` (overrides saved choices and
splits WSL galleries); a signed `.exe`, Electron or PyInstaller bundle.
