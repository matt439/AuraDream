# Review 2026-11-10: a first-time user, after the installer

Matt asked what AuraDream is like to install and use for the first time for
someone inexperienced. The persona: a capable NVIDIA gaming PC on Windows 11,
no experience of git, Python, a terminal or Hugging Face, arriving from the
GitHub page. Five read-only agents walked it, each through one lens: the
install as written, failure paths, the first launch and first image, learning
the app, and day 2 (with the delta from the
[2026-11-01 first-run review](review-2026-11-01-first-run.md)). Two drove a
fake server with `tools/browser.mjs` at 1400x900 and read every screenshot.
A skeptic per lens then tried to refute each finding: of 56 (one of them the
delta summary), one was refuted and 14 were narrowed; the skeptics added ten
the finders had missed. Merged, 28 findings remain below, each pointing to its
`TODO.md` row (F27 to an existing one).

Hands-on checks: `install.ps1 -DryRun` (5.7 s, nothing changed),
`backend.doctor` (2.1 s, every row PASS on this machine), `update.ps1
-DryRun`, `install.cmd` copied alone to an empty folder, and the fake
server's first visit with and without a Hugging Face sign-in. No real server,
no card, no download and no clean machine: the installer's real run is still
`DEV#28`'s unverified part. Line numbers are as of ab051f6.

## The answer

The install is now easy for a beginner. From the GitHub page to a first image
on an open checkpoint is about 20 actions with no typed command (Download
ZIP, Extract All, double-click `install.cmd`, about 5 Enter presses, Switch &
load, a prompt, Generate); the gated default adds the Hugging Face account,
licence and token, about 33-38 actions. The installer's numbered steps,
`[stop] ... What to do:` lines, driver/card/RAM checks before any download
and the HF_HOME question with each drive's free space are the strongest part
of the experience.

The first hour after it is not easy yet. The novice meets four problems: a
first download of about 33 GB that the page calls "a few minutes"
([F1](#f1-the-first-download-looks-frozen)), a start-up refusal that vanishes
with its window ([F3](#f3-a-start-up-refusal-vanishes-with-its-window)), a
closed first-visit panel that leads to a gated failure whose only fix is a
terminal command ([F4](#f4-closing-the-first-visit-panel-leads-to-a-gated-dead-end),
[F5](#f5-signing-in-later-needs-a-terminal-in-the-project-folder)), and two
different "light first" checkpoints
([F7](#f7-two-different-first-open-checkpoints)). Past that, the app is dense
and its words are an expert's, and there is no help reachable from the page.

### What is good

- Zero typed commands on the installer route; `install.cmd` bypasses the
  execution policy for its one script and keeps its window open at the end
  (`install.cmd:13,17`).
- Every stop names the fix and says re-running is safe
  (`install.ps1:72-83`); a driver under CUDA 13 gets NVIDIA's link and "You do
  not need the CUDA Toolkit" (`install.ps1:370`); a low-RAM machine is steered
  to Animagine (`install.ps1:414-417`).
- OneDrive, long and non-ASCII folders are warned about; a network share or a
  running AuraDream refuses (`install.ps1:326-355`).
- A ZIP folder becomes a Git checkout safely, differing files backed up first
  (`install.ps1:510-544`); Python 3.14 on the machine does no harm (the
  3.13 pin).
- The doctor is fast, read-only and plain (`backend/doctor.py:536-572`).
- A second launch opens the page; a busy port names `AURADREAM_PORT`
  (`backend/main.py:1403-1415`); Hub failures during a load are plain words
  with Retry load (`backend/main.py:220-230`).
- With no sign-in, the first visit preselects an open checkpoint and says
  why; "closing the panel saves nothing" is said in the panel
  (`web/model-switch.js:750-757`).
- Generation itself reassures: a sharpening preview, "step 6/8", time left,
  and "Enter a prompt first." on an empty box. The Models rows show download
  size, cache state and access per checkpoint.
- Day 2 is one double-click; `update.cmd` refuses while AuraDream runs,
  copies the database first and explains a refused pull.

### Since 2026-11-01

F1 (RAM figures), F2 (closing the panel), F4 and F6 are fixed; F3 (a preflight
and plain words) is fixed in substance: the doctor, the CUDA message, the
401/403 wording and the busy port. Two parts of F3 remain: the status bar's
free space still measures the outputs drive (`backend/main.py:932-937`,
[F10](#f10-a-dropped-or-full-download-is-library-words)), and README's "add
`bin\` to your user PATH" (`README.md:146`) still has no steps. New since then:
the sign-in menu of the newer `huggingface_hub`
([F6](#f6-the-sign-in-menu-differs-from-the-instructions)), the passcode's
gaps in `SECURITY.md` and the Linux service
([F14](#f14-day-two-docs-gaps), [F15](#f15-the-linux-service-opens-without-a-passcode-warning)),
and FLUX 3's unknown download size and Expand RAM
([F26](#f26-flux-3s-expand-gives-no-ram-warning)).

## Install and start

### F1 the first download looks frozen

After Switch & load or Generate on a checkpoint not yet downloaded, the page
says "Loading weights 0:12 - the first load of a session takes a few
minutes." (`web/viewer.js:643`, `web/job-stream.js:290`); the Switch & load
path shows only "Z-Image Turbo · loading…" with no clock. At 100 Mbit/s the
default's 33.7 GB takes about 45 minutes (`docs/setup.md:287-289`). The only
progress is Hugging Face's bar in the console (`install.ps1:770-771`), which
`--open` has just covered with the browser. `snapshot_download` runs with no
progress hook (`backend/backends/loading.py:232-244`), but the catalogue size
and the cache state are already known (`backend/model_info.py:475-497`).

First step: while a load's checkpoint is not fully cached, report the bytes
under its cache folder against the catalogue size in `/api/status` and show
"Downloading 12.4 of 33.7 GB" in the topbar and viewer; at least change the
words to say a first download can take most of an hour, and give Switch &
load the same clock. `GEN#58`.

### F2 the console fills with request lines

The docs send the user to the console for that progress, but uvicorn's access
log is on (`backend/main.py:1443-1450`, `access_log` left at its default), so
the page's 3-second status poll (`web/app.js:943`) prints a line each time
around Hugging Face's progress bars. `log_file.py:36-37` keeps them out of
the file only. First step: `access_log=False` unless the log level is DEBUG,
and one clear start line ("running at ...; keep this window open; Ctrl-C
stops it"). `DEV#41`.

### F3 a start-up refusal vanishes with its window

The shortcut targets `bin\auradream.cmd --open` (`install.ps1:739-744`),
which exits with the server's code and no pause (`bin/auradream.cmd:6-9`).
A busy port 7860 (A1111 and Forge default to it), a short passcode, or a `uv
run` sync that fails after an update (`update.ps1:260`) prints one line, and
the window closes. These refusals come before `log_file.install`
(`backend/main.py:1397-1418`), so `auradream.log` has nothing either, and
setup.md's "If something goes wrong" (`docs/setup.md:309-358`) assumes the
message was seen. Windows Terminal may keep the window open; classic conhost
does not. There is no `--port` flag (`backend/launch.py:238-372`).

First step: pause in `bin\auradream.cmd` on a non-zero exit other than
Ctrl-C's; install the log file before the port and passcode checks; a
troubleshooting line for "the window flashes and closes" (run the doctor).
`DEV#42`.

### F4 closing the first-visit panel leads to a gated dead end

With no sign-in, the open checkpoint is only a preselection in the page
(`backend/main.py:135-143`; by design in the
[2026-11-01 review](review-2026-11-01-first-run.md)). Close the panel (the
natural move for a pop-up), type a prompt and press Generate, and the server
loads the gated FLUX.1 Krea [dev]. After the wait, the failure card
(`web/status.js:521-553`) offers Open the model page and Retry load only; its
sentence ends "run `uv run hf auth login` once with a read token, then press
Retry load" (`backend/main.py:228`, `backend/hub_access.py:37`). Hands-on,
the reason sat at y=892 in a 900-pixel window. The server already computes
the open suggestion.

First step: when a gated load fails and no token is present, the card offers
"Use Z-Image Turbo instead" (open Models with the suggestion chosen) before
the sign-in route. `UI#44`.

### F5 signing in later needs a terminal in the project folder

The page's gated words, the doctor (`backend/doctor.py:520-523`) and the
installer's skip line (`install.ps1:698-705`) all say "run `uv run hf auth
login`" without saying where. A novice who never opened a terminal (the
installer and shortcut did everything) opens PowerShell in their home folder,
where the project's `hf` is not available. Only `docs/setup.md:252` says
"From the project folder". None says that running `install.cmd` again offers
the sign-in (`install.ps1:686-696`).

First step: each message says "or double-click install.cmd again and answer
y at the Hugging Face step"; where the command stays, name the folder (the
server knows its root) and that pasted input does not show. `DEV#43`.

### F6 the sign-in menu differs from the instructions

The installer has the user create and copy a Read token, then says "Paste it
below" (`install.ps1:711-716`); setup.md step 6 and `README.md:88-94` say the
same. The pinned `huggingface_hub` 1.33.0 (`uv.lock:498-499`) first shows an
arrow-key menu, "How would you like to log in?", with "Log in with your
browser" as the default (`huggingface_hub/_login.py:306-317`). Enter then
gives a URL, a code and "Waiting for authorization" to someone holding a
token; "Paste an access token" gives a hidden input. The browser route works
and needs no token.

First step: document the browser login as the route (drop the
create-and-copy step), keep pasting as the fallback with the hidden-input
note, find whether the device-flow token expires and say what to do, and pin
the menu's wording with a test against the installed library. `DEV#44`.

### F7 two different first open checkpoints

The installer recommends and first starts Animagine XL 4.0 ("a much smaller
download and lighter on memory", `install.ps1:41,764`, `docs/setup.md:221-225`)
and, in the same breath, says "32 GiB for the default is about 45 minutes"
(`install.ps1:769-771`). The shortcut passes only `--open`
(`install.ps1:741`), so a later start with no saved choice and no sign-in
preselects `OPEN_DEFAULT_MODEL_ID`, Z-Image Turbo
(`backend/model_defaults.py:156`): 32.8 GB, and 26.4-26.6 GB of process RAM
at the load (`docs/verified.md:1410-1413`). `docs/setup.md:281-285` names it
without size or RAM.

Matt chose Z-Image Turbo for `GEN#41` on 2026-11-03, so which open checkpoint
is first is his call: one of the two everywhere (installer, page, doctor,
docs), or the shortcut carrying the installer's model until a choice is
saved. The installer's download line names the model it starts. `GEN#59`.

### F8 an existing token skips the question

A token left by another AI tool (`HF_TOKEN` or the token file) makes step 10
say "[ok] Already signed in" and set the gated route without asking
(`install.ps1:686-691`); step 12 starts with no `--model` (`install.ps1:764`),
so the page stays on the 33.7 GB FLUX.1 Krea [dev], even after step 3 told a
machine under 40 GiB of RAM to start with Animagine. A token is not an
accepted licence either. First step: ask which to start on when a token is
present, defaulting to the open checkpoint when step 2 or 3 noted less than
measured. `DEV#45`.

### F9 out of memory is library words

On an 8-12 GB card, a FLUX-size checkpoint or a large size fails with
PyTorch's text, cut from the middle so that `PYTORCH_CUDA_ALLOC_CONF` stays
as its tail (`backend/error_text.py:21-28`, `backend/jobs.py:847-863`,
`backend/main.py:253`). The plain advice (a smaller size, Sequential offload)
is only in `docs/setup.md:353-354`. A Windows paging-file failure (WinError
1455, `MemoryError`) at the load is not mapped either. First step: map
`torch.OutOfMemoryError` on a load and a cell, and the RAM failures on a load,
to plain sentences; the raw text stays in the log. `GEN#60`.

### F10 a dropped or full download is library words

`_load_failure` gives words to the Hub's gated, denied, not-found and offline
kinds only (`backend/main.py:219-253`); a Wi-Fi drop or a full HF_HOME drive
partway through 33 GB shows `short_error`'s "ClassName: ..." and nothing says
Retry load keeps what is downloaded. The topbar's unlabelled "N GiB free" is
the outputs drive (`backend/main.py:932-937`, `web/status.js:885`), not the
cache drive the installer suggests moving, and Switch & load compares no
size with any free space (the grid preview does, `web/grid-preview.js:208`).
First step: classify ENOSPC and connection/timeout failures on a load; report
the cache drive's free space and warn in Models when a row's download exceeds
it. `GEN#61`.

## Learning the app

### F11 no help from the page

The topbar's `?` is "Keyboard shortcuts (?)" (`web/index.html:68`): 11 groups
starting with Ctrl/Cmd+Enter rules. Nothing in the page links to
`docs/setup.md`, `docs/usage.md` or the README. The per-field `?` hints are
real help but say nothing about getting started. `UI#43` (from the peer
review) serves usage.md from the shortcuts dialog. First step: a short
"Getting started" block at the top of that dialog (pick a model, a prompt,
Generate, where images go, how to stop) beside `UI#43`'s link. `UI#45`.

### F12 README's front and the docs' shape

Before the installer, README opens with phone use naming three `AURADREAM_*`
variables (`README.md:16-24`), then Requirements with "cu130", "compute
capability 7.5 to 12.0", "27.9 GiB working set and 35.4 GiB private" and
"**Python 3.13** and uv" (`README.md:38-66`; setup.md says not to install
Python), and contradicts itself on the measured cards (`README.md:39-43`
names the 8 GB RTX 3070, `:45-49` "the only card measured"). Quick start says
"download or clone" without extracting (`README.md:70-71`). usage.md (1,493
lines) has no FAQ or glossary; setup.md's first-picture walk
(`docs/setup.md:261-306`) is good, and its restart advice (`:298-304`) is
`cd AuraDream` in PowerShell rather than the shortcut.

First step: a one-line Windows route above Requirements, the Python bullet as
"uv (the installer brings Python)", the jargon behind a link, the cards
sentence made one, the phone paragraph below Quick start, restart by the
shortcut, and a short FAQ in usage.md (where are my images, why is the first
one slow, which model, my picture looks wrong). `DEV#46`.

### F13 phone use on Windows has no recipe

README pitches phones with screenshots and says "on Windows set
AURADREAM_HOST and AURADREAM_ALLOWED_HOSTS, and AURADREAM_PASSCODE", linking
to the CLI section; `docs/usage.md:1153-1157` adds one sentence. Nothing
covers setting a variable so the shortcut sees it, the PC's address
(`ipconfig`), the Firewall prompt, or the bare "Invalid host header" a missed
allowed host gets (`backend/main.py:620`); the start warns only about a
missing passcode (`:1434-1441`). First step: a "From a phone (Windows)"
section in setup.md, and a start warning when the bind is not loopback but
every allowed host is. `DEV#47`.

### F19 the first-visit panel's choice is out of sight

The auto-opened Models panel shows Quantization, Offload and a Performance
fold first (`web/index.html:77-94`), then 27 rows. Hands-on, the preselected
Z-Image row sat at y=1620 in a 900-pixel window, so no visible radio was
ticked and the visible top row was FLUX.1 Krea [dev] marked Current; the note
does name the choice. No row says "start here" (access and fit badges do
help). By design the suggestion is not scrolled to. First step: on the
automatic opening only, scroll the table (never the page) to the chosen row
and name it on the button ("Switch & load Z-Image Turbo"); consider folding
the loader options on a first visit. `UI#46`.

### F20 inert and premature controls

57 controls are visible at 1400x900 with an empty gallery, among them the
gallery's batch tools (Picked, Select, Swap tags, Refine, More seeds, Saved
searches, Bundles) before any image exists. Step cache offers only "FBC 0.05
v1 (unavailable)" (`web/index.html:833-836`) and Performance says "no other
choice is available yet" (`:90-94`). First step: hide the Step cache row and
the Performance line while no other choice is admitted, and the gallery's
batch row (its height reserved) until the gallery has an image. `UI#47`.

### F21 the tooltips are written for experts

The Sampling row is "Steps · CFG (distilled) · Mu · True CFG"; the tips say
"Denoising steps…", "Classifier-free guidance…", "Flow-matching timestep
shift…" (`web/index.html:467-505`, rewritten from the profile at runtime).
The defaults work (Mu auto, True CFG off), so it is low. First step: lead
each tip with one plain clause on the visible effect, keeping the sourced
detail after it; a short glossary in usage.md. `UI#48`.

### F22 Edit by instruction on every image

The right-click menu has 18 items and puts "Edit by instruction…" on every
image (`web/context-menu.js:204-205`), a FLUX.1 Krea one included, whose
dialog then explains the Kontext envelope. First step: group the pass actions
under one Edit entry, or show Edit by instruction only where switching to
Kontext is the user's evident aim; the lightbox's compare block as one
Compare… until there are neighbours. `UI#49`.

### F23 a never-started run's box stays after a switch

After the gated failure and a switch to Z-Image, the red last-run box under
Generate still read "0 images, never started ... Finish run: made on
black-forest-labs/FLUX.1-Krea-dev; the server is running
Tongyi-MAI/Z-Image-Turbo" (`backend/api_runs.py:127,274` use raw ids). First
step: offer no Finish run for a run with no image that failed at its load,
clear the box once another checkpoint loads, and use catalogue names.
`UI#50`.

### F24 Check leads with encoder internals

Check on "a red bicycle leaning on a brick wall at sunset" shows Encoders
("CLIP-L 10/75 one pooled vector…", "T5-XXL 10/511"), "Order: Not checked…",
one useful finding (under the guide's 30-80 words), a 4.5 GiB critic offer
and Compare (`web/index.html:286-345`). Check is an opt-in press, so low.
First step: the findings first in plain words, Encoders/Order/Compare in a
shut Details fold, the critic named by what it does. `PA#13`.

### F25 Examples' notes in warning amber

Each author example carries "Author sample: under 30-80 words" and the like
(`backend/guides.py:83-84,156,196`), drawn as `.example-note` in
`var(--warn)` (`web/prompt-examples.js:36-40`, `web/styles/fields.css:322`),
which reads as the official examples being wrong. First step: the hint
colour and a context phrasing. `PA#14`.

## Day 2

### F14 day-two docs gaps

- **One checkpoint cannot be removed by any documented route.** Each is 7-58 GB
  (`backend/model-downloads.json`); `models--` folders appear only under
  Uninstall (`docs/setup.md:428-434`). `GEN#55` is the storage view; until
  then a "Free disk space" paragraph (where each lives, the converted copies,
  `hf cache` scan and delete while stopped).
- **Uninstall can delete the backup.** Step 1 offers "save a bundle", which
  lands in `bundles/` beside the database, inside the folder step 2 deletes
  (`docs/setup.md:424-427`, `docs/usage.md:1303-1305`). Say to Download it,
  or copy `bundles/`, outside first.
- **`SECURITY.md` still says "no login"** (`SECURITY.md:3-9`) since the
  passcode; add it, its plain-http limit and gate bypasses in scope.
- **There is no Getting help section.** It would say where to ask, and what to
  attach: the doctor's output, the Commit row, the log lines (read first:
  they hold prompts). An issue template under `.github/` needs the ruleset
  edit (`ci-definition`), so it is optional.

`DEV#48`.

### F15 the Linux service opens without a passcode warning

`linux/install-service.sh` writes `AURADREAM_HOST=0.0.0.0` with
`#AURADREAM_PASSCODE=` commented (`:130-143`), starts the service and ends
with the LAN address (`:281-300`); the "no passcode" warning is a log line
(`backend/main.py:1434-1441`). First step: a `[warn]` at the end when the env
file has no active passcode, naming the file and the restart. `DEV#49`.

### F16 WSL setup says Ready without CUDA

`wsl/setup.sh` checks only that `nvidia-smi` runs (`:16-20`), prints "cuda
MISSING" and carries on to "Ready." (`:42-55,96-101`); it fetches uv's
installer unpinned (`:29`), and on an RTX 20-40 card names "the int4 repos"
without `AURADREAM_NUNCHAKU_FORMAT=int4` (only `README.md:190-192` does).
`linux/setup.sh` stops on a False CUDA check and runs the doctor. First step:
the same stop, the driver's CUDA version, the doctor, uv pinned, and the
export line printed. `DEV#50`.

### F17 installer wording

All low, one row:
- The VRAM note says only a 16 GB card was measured (`install.ps1:400`); the
  8 GB RTX 3070's Animagine result is in setup.md.
- "[ok] No problems found." follows the doctor's WARN lines
  (`install.ps1:666-668`), the Hugging Face one being the next step's
  question.
- `install.cmd -DryRun` and `update.cmd -DryRun` need `.\` in PowerShell
  (`docs/setup.md:81-82,366`).
- `install.cmd` run from inside the ZIP gives PowerShell's raw `-File` error
  (`install.cmd:13`; checked hands-on); say to extract first.
- An AMD or Intel PC is told to install the NVIDIA driver
  (`install.ps1:361-362`).
- Nothing lifts the downloaded-file mark from a ZIP folder, while README says
  the `.ps1` launchers run under RemoteSigned (`README.md:150-151`).
- The Git-checkout question (default yes, `install.ps1:491-496`) does not say
  what Git is for.
- `bin\auradream.cmd` has no fallback when uv is not on PATH.

`DEV#51`.

### F18 the update says up to date either way

`update.ps1:249` prints "The code is up to date." after any successful pull;
there is no version, tag or changelog (`pyproject.toml` 0.1.0, `DEV#29`
deferred), and a blocking `.venv` process is named by its PID only. First
step: "Updated from <old> to <new> (N changes)" or "Already up to date", and
the blocking process's command line. `DEV#52`.

### F26 FLUX 3's Expand gives no RAM warning

Klein loads in about 19-21 GiB, so it looks right for a 32 GB PC, but its
upsampler's first Expand needed about 41 GiB of process RAM
(`docs/verified.md:2104-2108`); the page's cost line names only the download
(`web/status.js:261-268`). First step: the measured RAM on FLUX 3's Expand
cost line. `GEN#62`.

### F27 FLUX 3's download size is unknown

Both FLUX 3 rows show an Unknown download and cache state
(`backend/model-downloads.json`, `files: null`); setup.md gives Klein's memory
but no size. This is `GEN#46`'s shard-map step; no new row.

### F28 docs/ is mostly maintainer records

Of 108 files in `docs/`, 95 are reviews, plans, checklists or findings, which
GitHub lists before rendering `docs/README.md`'s "Using the app". README's
navigation line links setup.md and usage.md directly, so the cost is small.
First step: a `docs/maintainer/` move with every citation updated, only if
Matt wants the churn. `DEV#53`.

## Not taken

- **The non-ASCII user profile** (the installer checks only the project
  folder's path): refuted. No tool in the chain was shown to break on one, and
  uv, uv's Python and the Hub cache handle Unicode profiles.
- **A RAM fit badge in Models**: folded into [F7](#f7-two-different-first-open-checkpoints)'s
  decision rather than a badge of its own; RAM at the load is measured for a
  few checkpoints only.
