# Review 2026-11-10: the app against comparable projects

Matt asked for a review of the project for improvements and new features,
with similar projects as the yardstick. Twenty-five read-only agents ran it:
five mapped the app (capabilities, the recorded decisions, backend, frontend,
and operations and security), six researched peers (ComfyUI; Forge,
AUTOMATIC1111 and SD.Next; InvokeAI; Fooocus and SwarmUI; Krita AI Diffusion,
Draw Things and others; current models and techniques), three derived gaps,
and eight skeptics checked every candidate against the code, `TODO.md` and
the earlier reviews. Of 79 raw candidates, 48 remained after merging and 23
were kept. None was rated high value.

The checks were made by reading. No server, test, smoke or card run was
made, so each finding's behaviour is confirmed in the source only; the line
numbers are the tree's on 2026-11-10. Each finding below owns its `TODO.md`
row: what is wrong, the first step, and what the skeptic cut from the
original proposal.

## Against the peers

The widest gaps are ones already decided, and are not reopened here:
ControlNet, IP-Adapter and regional prompting (absent; ControlNet recorded
as not taken, "nothing in the ratings asks for them"), a standalone upscaler
library (enlarging through Refine covers it), LoRA stacking and a checkpoint
grid axis (the tasks are rated Never or cut against the switch rules), and a
project taxonomy.

Two peer features survived, both under Refine and enlarge, which
[reported use](../design/preferences.md) rates Often: [R1](#r1-refine-on-the-families-that-have-none)
and [R2](#r2-tiled-refine).

## Matt's answers (2026-11-10)

- **A passcode for the LAN service: yes.** [S2](#s2-passcode-beyond-loopback) is ready work.
- **Which non-refining checkpoint to enlarge from:** initially not sure;
  **Anima selected 2026-11-11**. [R1](#r1-refine-on-the-families-that-have-none)
  now has a CPU feasibility result and implementation plan.
- Every kept finding gets a row.

## Editing and generation

### R1 refine on the families that have none

`GEN#48`. `refine_supported` is True only where diffusers ships an
image-to-image sibling (`refine.sibling`): FLUX.1, SDXL, Z-Image and
Qwen-Image. Krea 2 (both profiles), Anima, FLUX.2 klein and FLUX 3 Klein
have no Refine; the docs state it as a fact, not a decision. Klein's inpaint
is already `INP#5`. The new part is an app-owned image-to-image entry:
encode the source, noise it to the start sigma, denoise the tail. Today's
pass leans wholly on diffusers classes, so this is a new pattern, one family
first, and it needs the card.

**Decision 2026-11-11: Anima first.** The exact installed diffusers
0.42.1 already has Anima image-to-image blocks. The cold inspection and
CPU schedule/noise probe, remaining integration work and separate card
qualification are owned by [the Anima Refine plan](plan-anima-refine.md).
Selecting the family does not open Refine admission or establish its pixels,
fit or quality.

### R2 tiled refine

`GEN#49`. Refine is one whole-image pass under per-profile size guards; the
only tiling is the VAE's and the opt-in Anime6B step's. A tiled pass would
run the existing image-to-image per overlapping tile, with tile size,
overlap and blend frozen and keyed as `mask_crop.region` is. It helps the
8 GB server most. Per-tile denoising with a pixel feather is not the peers'
multi-diffusion (which blends latents every step), so seams or drifting
content are likely at useful strengths.

First step, on the card with Matt's OK: refine one 2x FLUX.1 or SDXL source
as four overlapping Only the masked area regions at strength 0.3-0.4 and
look at the seams. A plan is written only if that is acceptable.

### R3 a wider instruction-edit envelope

`INP#13`, after `INP#11`'s card acceptance. Any 1:1 source qualifies today
and renders at 512x512; output is capped at 262,144 pixels and other aspects
are held. The [plan](plan-instruction-editing.md) already freezes six
aspects under the 512² area, each opened by one `QUALIFIED_ENVELOPE` line.
A 1024x1024 edit is a new frozen recipe with its own name and card
measurement (stage 3 at 512² took free RAM from 42.5 to 15.2 GiB). Recasting
klein's reference prompts as an `edit` spec, and CLI parity, were cut.

First step: a plan addendum defining the second recipe (1024x1024, 1:1
only) and a one-load card protocol; open the frozen aspects in the same
session.

### R4 release the card after idle minutes

**Built 2026-11-11 (`GEN#50`).** The design note fixes one default-off
setting, `AURADREAM_IDLE_UNLOAD_MINUTES=0`, and a 15-second event-loop
timer sharing KeepAwake's real-request clock, including when the power
inhibitor is off. Every live job/card/load/switch/bundle/counted worker or
model request holds it; work gets a full idle interval afterwards. The
ordinary unload now owns and shields its cleanup task, reserves the
selection until its workers finish, and drains at shutdown even if a
request disconnects. The idle reason is assigned inside that reservation,
so a later Load clears it authoritatively. Status/page name an idle release.
Virtual-clock and fake/API tests cover activity, holds, cancellation,
cleanup and reload. No GPU, RAM-reclamation or sleep measurement is claimed.

`GEN#50`. Nothing unloads by itself. `KeepAwake` already tracks the last
real activity with timed polls excluded, and unload is already 409 while a
job or load is live. Opt-in and default off (manual Unload is rated Never).
It frees the card, not RAM: the process keeps about 17 GiB afterwards. On
the 3070 server it unloads before the sleep hold lapses, which sidesteps
`CARD#14`'s suspend with a model loaded.

First step: a design note fixing one setting (`AURADREAM_IDLE_UNLOAD_MINUTES`,
0 off), the one place it fires (an event-loop timer calling the existing
unload) and the status wording; tested on the fake backend.

### R5 keep-awake on Windows

`GEN#51`. `keep_awake.platform_inhibitor` returns `NoInhibitor` off Linux,
while the 5080 PC is now driven from a phone with no local input. The
`Holder` abstraction exists, so a Windows holder (a power request handle)
sits behind the same providers, idle minutes, `counts_as_activity`
exclusions, status field and doctor row. A done sound and an action when the
queue empties were cut: a tone does not play on a locked phone, and a
command hook is a new trust surface on a LAN server.

Built 2026-11-10. The finding held in the source: `platform_inhibitor`
returned `NoInhibitor` for every platform but Linux, and the doctor's Keep
awake row returned nothing there. `keep_awake.PowerRequest` (kind
`power-request`) is now Windows' holder: a `PowerCreateRequest` handle with
`PowerRequestSystemRequired` set, cleared and closed on release, and closed
by Windows if the process dies. It is a handle and not
`SetThreadExecutionState` because the controller holds and releases on
different threads. It asks for the system only, so the display may turn
off, a sleep someone asks for still happens, and no power setting is read
or changed. Providers, counted holds, the idle minutes and
`counts_as_activity` are untouched; `/api/status`'s `keep_awake.kind` gains
`power-request`, and the doctor's row on Windows tries a request and lets
go at once. The CLI's `hold_while` takes the same hold. Tests fake the four
Win32 calls (`tests/test_keep_awake.py`); one makes a real request on
Windows and releases it in the same test, which passed on this PC. Not
built: the done sound and the queue-empty action, which were cut. Not
measured: that the PC stays awake past its sleep timer while the request is
held, which needs the sleep timer to run out on the card's PC
([design notes](../design-notes.md), Keeping the Windows PC awake).

### R6 a full disk pauses the job

`GEN#52`. `jobs._save` retries only on `GalleryBusy`. Any other error from
`storage.save_image` is a cell failure, and the job stops after three in a
row, so a full disk costs up to three renders. Saved images are kept and
Finish run resumes the rest. Fix: treat `ENOSPC` and SQLite's "database or
disk is full" like the busy branch, holding the render in memory and pausing
with a plain-words reason. A submit-time space check was cut.

First step: a CPU test making `save_image` raise each error, to pin today's
behaviour (the failure count, a leftover `.png.part`) before the pause.

Built 2026-11-10. The first step confirmed the finding on the CPU: with
`save_image` raising either error, a five-cell job rendered three cells,
failed each and ended as an error; `save_image` itself left no `.png.part`
(its own cleanup takes the partial and the thumbnail back), so there was no
leftover to fix. `jobs._save` now reads the error through `jobs.disk_full`
and pauses in `_hold` with `Job.pause_reason`, the render held and saved
at Resume; a Resume with the disk still full pauses again, a cancel ends
the job. Every other save error is still the cell's failure
(`tests/test_jobs_disk_full.py`; `smoke-jobs`' last steps through the
fake's `/api/fake/disk-full`). The tests also cover a batch whose second
image is refused, a Then refine pass's save, and SQLite's real error from
a database capped at its pages. The page takes the worker's own pause over
a Resume press still awaiting its reply (`syncPause`), or the button read
Pause over a job paused again. Not built: the CLI's own save loop
(`generate.py`), which still fails the cell (`GEN#57`, a row at Matt's
word 2026-11-10: hold the image and wait or stop with the same reason,
through `jobs.disk_full`), and the submit-time check that was cut. A real full disk has not been tried on the card's PC.

Built 2026-11-10 (`GEN#57`). The CPU test first showed the CLI did worse
than fail the cell: its save sits outside the per-cell `try`, so `ENOSPC`
or SQLite's error reached the outer handler, which threw the render away
and ended the run with exit 1, every later cell unrendered. The choice
between waiting and stopping went to waiting, as the CLI already does for
a busy gallery: `generate._save_held` reads the error through
`jobs.disk_full`, says once on stderr "the disk is full: this image is
held in memory. Free some space and it is saved; Ctrl-C stops without it",
and tries the save again every `DISK_FULL_POLL` (5 s) until it lands, so
an unattended night carries on by itself once there is room. The first
Ctrl-C ends the wait at once, saying "stopped with the disk full: run
2/3's image was not saved; --resume continues it", and the run ends as any
cancel does, exit 130 with the saved images kept; nothing is counted as a
failed cell. Then refine's passes save through the same helper, which also
gives them the busy-gallery wait they lacked; a pass stopped there says
"the pass of #12 was not saved; refine that image to make it", since
`--resume` skips a cell whose base image is saved. Every other save error ends
the run as before (`tests/test_cli_disk_full.py`). The wait has no limit
of its own and `--budget` does not end it (the budget only stops cells
from starting), and `--via-server` is unchanged: the server's job pauses
and the CLI goes on watching it. A real full disk has not been tried here
either.

### R7 prompt recall

`GEN#53`. A prompt is recalled only through its image (Copy prompt, Reuse,
search). First slice: Use prompt beside Use seed in the context menu,
writing prompt and negative through `rewriteBox`. Only if that falls short:
Recent prompts under Examples (no panel height), a model-free read of
distinct recent pairs. A gallery-derived list cannot hold prompts whose jobs
saved nothing.

Built 2026-11-10, the first slice only. The finding held: the menu had Copy
prompt and Reuse settings and nothing that put a prompt back by itself.
**Use prompt** follows Use seed in the image's menu - a card, a result, a
sheet's cell, the lightbox and the phone's More sheet, which is the same
list. `usePrompt` (reuse.js) writes the record's prompt and negative
prompt through `rewriteBox` and leaves the seed, size and sampling alone.
A record with no negative empties the negative box, as Reuse does. Where
either box held other text the note under the prompt offers an Undo for
both, and Ctrl+Z works in each box. From the lightbox it closes the
lightbox first, as Reuse does: a modal keeps the focus the write needs.
It is held on an instruction edit, whose words are about its source
(`tests/test_use_prompt.py`; `smoke-context-menu`, whose `execCommand`
stub now lets a typing write through; `smoke-phone-gallery` chooses it in
the phone's More sheet). Not built: Recent prompts under
Examples, kept for when this falls short; a prompt whose job saved nothing
is still not recalled.

### R8 a snippet target for grid variables

`GEN#54`. Snippets target Prompt or Negative only
(`schemas_snippets.py`); a variable block persists only inside a saved
setup. A third target appends a saved `name = a | b` block to the grid's
variables box as Append does today, leaving expansion and reproduction
untouched. Server-side resolution of a bare `{name}` and style templates
with a `{prompt}` slot were cut. Starts as an addendum to
[the snippets plan](plan-prompt-snippets.md).

Built 2026-11-10, after [the addendum](plan-prompt-snippets.md#addendum-2026-11-10-a-third-target-the-grids-variables).
The finding held: the target was `Literal["prompt", "negative"]` in three
places, and nothing but a saved setup kept a variable block. The three now
share `schemas_snippets.SnippetTarget`, which adds `variables`; the sidecar
and the portable document stay version 1, and export, preview and import
carry the target as they carry the other two (names collide per target, a
Replace offers only a variables snippet). The text is stored verbatim and
never parsed. In the dialog **Insert into** offers **Grid variables**:
Insert closes the dialog, opens the Grid search fold (More options first
on the phone), and appends or prepends the block with one newline through
`rewriteBox`, so Ctrl+Z restores the box; the grid's own preview then reads
it and refuses what it always refused. Expansion, keys and records are
untouched (`tests/test_snippets.py`, `tests/test_snippet_portability.py`,
`tests/test_api_snippet_portability.py`; `smoke-snippets`' last steps, and
`smoke-phone-dialogs` for the phone's sheet). A build from before this
refuses a sidecar or an export file holding a variables snippet whole and
leaves it unchanged. Not built, as cut: a bare `{name}` resolved on the
server, and `{prompt}` style templates; the `{name}` placeholder is still
typed into the prompt by hand.

### R9 storage view

**Built 2026-11-11 (`GEN#55`, first slice).**
`GET /api/models/storage` explicitly starts one daemon-thread inventory,
with a reservation held through a disconnected caller's worker. It maps
retained source revisions/file selections to every catalogue entry and
reports overlapping/shared/exclusive file bytes, converted copies,
incomplete downloads, free space and unreferenced revisions/repositories.
Unknown selections have no claimed model total. Scratch-cache and API
tests check shared accounting, unknown/corrupt/missing caches, concurrency
and no mutation. "Unreferenced" means absent from these retained catalogue
references: the critic, other helpers and other apps can need those files.
This is neither disk-allocation accounting nor a deletion plan. The guarded
delete remains a later design after this inventory has been considered.

**Matt's look, 2026-11-14 (`GEN#68` closed).** Models > On disk over the
real Windows cache "looks ok": a 551.2 GiB Hub cache with 926.7 GiB free,
48.3 GiB of converted copies, 17.5 GiB not referenced, no incomplete
downloads, and 16 catalogue rows (Anima's three sharing 1.4 GiB, three
SDXL entries 3.0 MiB). One row is a bare id: Krea 2 Turbo's Nunchaku
transformer (`lite-infer/...nvfp4_r32-bnb4-text-encoder`, 10.9 MiB, a
quantization's source rather than a catalogue entry), which the delete's
design should name by its entry. Matt's answer on what he would delete:
any model, for any of these reasons: he no longer uses it, it is
outdated, he needs the space, or its licence has expired. So the guarded
delete (`GEN#72`) takes a whole checkpoint, not only stale revisions or
unreferenced bytes; its design must say what happens to files another
entry shares, to its converted copies, to the loaded or selected
checkpoint, and to a job or load in flight.

`GEN#55`. Models shows each entry's reference download size and a cache
word, not bytes on disk. Absent: measured bytes per entry, the cache total,
free space on the cache drive, any delete, and what no entry references
(old revisions, the critic's GGUF, FLUX 3's 18.8 GB upsampler). First slice:
a read-only, on-demand route running `scan_cache_dir` on a daemon thread,
mapped to catalogue entries through the download references. The guarded,
idle-only delete is designed after that output is seen.

### R10 capability flags per checkpoint

`GEN#56`. The Models picker shows Style, Refine, Size, Speed and notes, but
not inpaint, outpaint, references, LoRA, negative prompt, live preview or
Expand, and no doc holds one table across the catalogue. Add the missing
flags to `model_info.details_for`'s notes (no new columns) with a test over
every catalogue entry, and generate a `docs/usage.md` table from the same
function, held equal by a test.

Built 2026-11-10. The finding held: `details_for` had no word for any of
the seven, and Expand's rule by id lived inside the backend.
`model_info.capabilities_for` answers each flag as Yes, No or Not yet (an
inpaint class diffusers ships and the app has not enabled) from the place
that decides it at run time: the profile for inpaint, outpaint, references
and the negative prompt, `loading.lora_refusal` and the Nunchaku rule for
LoRA, `latent_preview`'s fitted maps for the preview (through `_LATENTS`,
each entry's latents by exact id) and `loading.prompt_expander_by_id` for
Expand, which the backend and the fake now call too. `details_for` says
them as `capabilities_note`, and the page joins that to the Refine note:
the Refine cell's tooltip and the radio's description, no new column.
docs/usage.md has "What each checkpoint can do", a table generated by
`tools/capability_table.py --write` from `model_info.capability_table()`
(`tests/test_capabilities.py`; `smoke-model-hover` and
`smoke-model-switch` hold the note in the page). Refine is the table's
first column, from `details_for`. Not built: a column for Edit by
instruction, whose answer is the backend's live one, not the id's; the
row's Edit note says it. Not settled: a change to usage.md alone runs no
tests, so a hand edit to the table is caught at the next code change;
`tests/test_ci_changes.py` names the exception, and naming the file in
`tools/ci_changes.py`'s `_TESTED_PROSE` is a CI-definition change for
Matt's ruleset edit. The preview words are the code's rule for each VAE,
seen on the card only for the checkpoints docs/verified.md names.

## Server and security

### S1 anti-framing headers

`DEV#31`. No response carries `Content-Security-Policy`,
`X-Frame-Options` or `X-Content-Type-Options`. A cross-site page framing the
app gets the app's own origin inside the frame, so framed writes pass
`block_cross_site_writes` and the Host check. Severity is low (blind clicks
at a guessed address; bins are recoverable). Fix: one middleware adding
`frame-ancestors 'self'` and `nosniff` to every response, mounts included;
`'self'` stays because `smoke-jobs` frames `/`. A full `script-src` policy
is a separate, larger item and not opened.

Built 2026-11-10: `backend/response_headers.py`'s
`ResponseHeadersMiddleware`, a plain ASGI layer `main` adds last, puts
`Content-Security-Policy: frame-ancestors 'self'` and
`X-Content-Type-Options: nosniff` on every response: the routes, both
mounts, `/sw.js`, `/docs`, and the Host check's 400 and the cross-site 403
under it. The module also pins `.js` and `.css` to their types, since
nosniff refuses a script served as `text/plain` and Windows' registry can
say that. Starlette's own 500 for an unhandled error comes from a layer
outside every one an app can add, so the layer answers such an error itself
in the same words and raises it on for the server to log. No
`X-Frame-Options` and no `script-src`. The three smokes that frame `/`
still pass (`tests/test_response_headers.py`; rule in `docs/rules.md`
under Frontend and CLI).

### S2 passcode beyond loopback

`DEV#32`, approved by Matt 2026-11-10. There is no authentication:
`TrustedHostMiddleware` and the cross-site write block only, with
`/docs` and `/openapi.json` served, and the Linux service bound to the LAN.
An opt-in passcode (`AURADREAM_PASSCODE`) checked in middleware, off on
loopback-only binds. It must cover the API, SSE, the `/outputs` and
`/static` mounts, `/sw.js` and `offline.html`, the `--via-server` client and
the fake server, and needs a small sign-in page. Over plain http the
passcode and its cookie travel in clear text, so the docs say it keeps out
the household, not an attacker on the wire; [S3](#s3-an-https-route) is what
makes it private.

First step: a design note listing every unauthenticated surface before any
code.

Design note written 2026-11-10, no code yet:
[A passcode beyond loopback](../design-notes.md#a-passcode-beyond-loopback)
lists the surfaces (88 API operations on 77 paths, the stream, both mounts,
`/sw.js`, four documentation routes, and every script and tool that calls
the server) and fixes the design and a four-stage build order. The code is
next, from its stage 1. Reading the source confirmed the finding. Two
points go past its wording and are Matt's to overrule before stage 1: the
passcode is also asked on a loopback bind whose `AURADREAM_ALLOWED_HOSTS`
lists another name (a proxy in front), and a request from 127.0.0.1 is not
exempt while it is on, so the scripts on the server send the passcode from
the environment file (where, since bash and systemd both read it, the
passcode is limited to letters, digits and a few plain marks). The limit on wrong passcodes is for the whole server, so while
someone keeps guessing, a right passcode without a cookie is refused too.
`sw.js` and `offline.html` stay behind the passcode
and need no change; only the manifest and its icons stay open.

Matt read those flagged choices on 2026-11-10 and said go to all of them as
written (asked behind a proxy too, no exemption for 127.0.0.1, the
environment file's character limit, one whole-server limiter, the
8-character floor and the 30-day stateless cookie). Stage 1, the gate, is
built the same day: `backend/passcode.py`, the setting, the layer in
`main.py` with the model hold moved inside the Host check and the write
block, `web/passcode.html`, `passcode` in the status, and `main.run`'s
refusal and log lines, with a test that walks the app's route table so a
route added later is closed or the test fails. What stage 1 settled is in
the note under [Stage 1 as built](../design-notes.md#stage-1-as-built).

Stage 2, the callers, is built the same day. Everything that is not a
browser reads `AURADREAM_PASSCODE` from its own environment and sends it as
`Authorization: Bearer`, never as an argument: `generate --via-server`, the
CLI's check for a server holding the card (a caller the note's table had
missed), the doctor's Commit row, and the Linux scripts' busy and status
reads (`linux/common.sh`, `linux/control.sh`), which take it from the
service's environment file. A 401 and a 429 are each said in their own
words and never read as a server that is down or idle: stop, restart and
`update.sh --restart` refuse without `--force`. The doctor gains a Passcode
row. The scripts refuse an environment file whose passcode bash would read
differently from systemd before they source it, and `install-service.sh`'s
new file names the setting in a commented line. The title probes needed
nothing. `tests/test_linux_scripts.py` was run under WSL. Still to build:
the page (the signed-out notice, Sign out), the fake's knob, the smoke and
the docs; until then the setting is not named in the README. What stage 2
settled is under [Stage 2 as built](../design-notes.md#stage-2-as-built).

Built 2026-11-10, stages 1 to 3. Stage 3, the page, the same day: a session
that ends under an open page shows "signed out" in the top bar with a Sign
in button and never reloads by itself, so the prompt is kept; Sign out is
in the Models panel while a passcode is asked; the fake server has a
`/api/fake/passcode` knob; `smoke-passcode` walks the sign-in page, a wrong
passcode, the lock, the app opening, an image and a job stream, the ended
session, Sign out and the sign-in page under the phone audit, and
`smoke-offline` ends with the gate on. The README, the usage guide and the
setup guide now name `AURADREAM_PASSCODE` and say what plain http leaves
readable: it keeps out the household, not anyone who can capture the
network's traffic. What stage 3 settled is under
[Stage 3 as built](../design-notes.md#stage-3-as-built).

**Stage 4, the real machines, is Matt's and untried.** Everything above
was checked with tests, a fake server and a headless browser on one PC.
Nobody has yet set a passcode on the Linux service and signed in from a
phone or another PC, tried Safari's password autofill, the home-screen
icon, a session across a restart or an update, `auradream-server status`
and `update.sh` through the passcode, or a wrong passcode's lock from a
second device. Until that is done this finding is built, not proven.

### S3 an HTTPS route

**Built 2026-11-11 (`DEV#33`).** `AURADREAM_ALLOWED_ORIGINS` adds exact
scheme/host/effective-port origins for a proxy on another port, while the
independent Host allowlist remains required. Credentials, paths, fragments,
queries and wildcards are refused. The optional PEM certificate/key pair
is passed to uvicorn; startup, local probes, CLI, doctor and Linux controls
choose HTTPS and retain certificate verification. An unverified configured
listener refuses installer/update work. The remote launcher's optional
`AURADREAM_SERVER_URL` chooses its browser address separately from SSH.
Setup documents a loopback-backed Caddy recipe, client CA trust, direct TLS
and their explicit setup boundaries. A real loopback TLS test uses a
throwaway certificate trusted only in the test process; no system trust,
firewall, proxy, service or real-phone setup was changed.

`DEV#33`. `main._origin_allowed` requires the Origin's port to equal
`settings.port`, so a TLS proxy on 443 has every write refused;
`docs/usage.md` documents "serve it on the same port". Without a secure
context the phone loses Share, the copy items, Notify when done and the
service worker. Fix: a public-port or allowed-origins setting read by the
origin check, optional certificate and key passed to uvicorn, and a
documented recipe. Standalone PWA is separate and not opened.

### S4 the before-update copy

`DEV#34`. `update.ps1` and `linux/update.sh` copy the database to one fixed
`<stem>-before-update.db` before every pull. A second update, after the new
version has migrated, replaces the only pre-migration copy. Fix: name the
copy by the pre-pull short commit, leave an existing copy for the same
commit alone, keep the newest three, and write the way back in
`docs/setup.md`.

Built 2026-11-10. Both scripts name the copy
`<stem>-before-update-<commit>.db` from `git rev-parse --short HEAD` read
before the pull, leave a copy already there for that commit as it is, and
delete all but the newest three commit-named copies beside the database
(by modification time, this update's never among the deleted, each with its
`-wal`/`-shm`). The old fixed name is used only when git answers no commit,
and an existing fixed-name copy is never counted or deleted. A dry run
prints the copy, the copy it would leave and each delete. The claims held
as written: both scripts copied with `-Force`/`cp -f` to the one name.
`docs/setup.md` has "Going back after an update"; the rule is in
`docs/rules.md` beside the installer's. Tests:
`tests/test_install_scripts.py` and `tests/test_linux_scripts.py`, the
second update keeping the first copy among them. One consequence of the
rule as written: a copy left alone is not refreshed, so after an update
that pulled nothing, the next update from that same commit keeps the
earlier copy, without the images made in between (their PNGs remain, and
`backend.reindex` reads them back).

### S5 a log file

`DEV#35`. Logging is `basicConfig` to stderr; job, cell, load and switch
failures log their traceback only at DEBUG, and uvicorn's level is
hard-coded. On Windows a failed overnight job leaves nothing once the
console closes. Fix: a rotating file beside the database that always
receives tracebacks, `faulthandler` into it, uvicorn following the
configured level; the console unchanged. The file holds prompts and paths,
so it is never bundled, exported or treated as gallery data, and is
installed in `run`, not at import.

Built 2026-11-10: `backend/log_file.py` writes `auradream.log` beside the
database (5 MB, three older files), installed by `main.run` after the port
checks and closed when the server stops. It receives the configured level's
records and, at every level, any record with a traceback, which covers the
job, cell, pass, load and switch sites unchanged; the `auradream` and
`backend` loggers are opened to DEBUG for that and the console's handlers
held at the configured level. `faulthandler` writes to the same file, kept
on it across rollovers, unless it was already enabled. uvicorn takes
`AURADREAM_LOG_LEVEL`, and its `uvicorn` logger (an unhandled route's
traceback) is followed; `uvicorn.access` is not. The rollover moves the
live file first, so one refused while another program holds the file open
keeps the older files. Tests in
`tests/test_log_file.py`; the reasons in `docs/design-notes.md` under "The
log file".

### S6 the running commit

`DEV#36`. The version is a static `0.1.0` and there are no tags.
`auradream-server status` prints the checkout's commit, which differs from
the running process after a pull without a restart. Read the short commit
once at start-up (None when there is no git), log it, and add it to
`/api/status` and the doctor. An in-page Machine check was cut to optional.

Built 2026-11-10. `backend/build_info.py` asks `git rev-parse --short HEAD`
in the project folder (None without a `.git` there, without git, or on any
answer that is not a commit). `main` reads it once at import as `COMMIT`,
beside `STARTED_AT`; `run` logs "AuraDream at commit ..." and
`/api/status` carries it as `commit`. The doctor's new **Commit** row
prints this folder's commit and, when AuraDream answers on the port, the
running server's own (`server_probe.running_commit`): the two differing is
a WARN that says to restart. `linux/control.sh status`, which
`auradream-server status` runs, prints the running server's commit under
"It answers" and says so when the checkout has moved on. The page shows
nothing new, and the in-page Machine check stays cut. Tests:
`tests/test_build_info.py`, `test_doctor.py`, `test_server_start.py`,
`test_api.py` and `test_linux_scripts.py`.

### S7 error codes the page reads

`DEV#37`. Over twenty sites in `web/` test English server wording
(`/unknown image/`, `/recycle bin/`, `/^\d+ batch/`, the LoRA's name) in
`err.message` to decide whether to forget or bin an id. A reworded refusal
would silently stop that cleanup. Fix: a stable `code` beside `detail` on
the few refusals the page acts on, the page reading the code. Response
models on every route and an OpenAPI snapshot were cut.

First step: list each `err.message` test in `web/`, the server sentence it
depends on and whether a test already pins it.

Built 2026-11-10. The list, as the source had it (line numbers the tree's
before the change); "pinned" means a pytest or a smoke asserted the
sentence, none asserted that the page depended on it:

| Page test | Sites | Server sentence | Pinned | Code now |
|---|---|---|---|---|
| `/unknown image/` | 18: `bin.js` 417, `facts.js` 223 and 428, `gallery-actions.js` 63 and 861, `inpaint.js` 890, `instruction-edit.js` 553 and 584, `lightbox-compare.js` 293 and 351, `lightbox.js` 833, `outpaint.js` 981, `reuse.js` 377 and 463, `sheet.js` 433, `sizes.js` 585 | "unknown image" (`api_gallery`: the record, the thumb, the star, the permanent delete) | `test_api_gallery.py` (the record only) | `image_unknown`, `image_binned` |
| `/recycle bin/` | 5: `facts.js` 224, `gallery-actions.js` 862, `sizes.js` 585, `outpaint.js` 170, `generation.js` 707 | "unknown image (#N is in the recycle bin)"; the presets' "no gallery image with id N (#N is in the recycle bin)"; Re-render's "in the recycle bin" | `test_api_gallery.py`, `test_api_outpaint.py` | `image_binned`; `RerenderGone.binned` |
| `/^no gallery image with id/` | `outpaint.js` 167 | the outpaint presets' 404 | `test_api_outpaint.py` | `image_unknown`, `image_binned` |
| `/^no gallery image with id ([\d, ]+)$/`, `/^reference image ... in the recycle bin; /` | `generation.js` 531 | `storage.reference_paths` through a submission | `test_api_conditioning.py`, `test_storage.py` | `image_unknown`, `image_binned` with `ids` |
| `/^no gallery image with id (\d+)\.?$/`, `/^source image (\d+) is in the recycle bin; /i` | `refine-rules.js` 135 (`sourceGoneKind`: Refine, Inpaint, Outpaint, Edit by instruction, their previews, the mask upload) | the same, and `POST /api/masks` | `test_api_refine.py`, `test_api_inpaint.py`, `smoke-refine` | the same |
| `/^Instruction-edit source #(\d+) is missing or in the recycle bin\.?$/` | `instruction-edit.js` 208 | `instruction_edit_admission._prove` | `test_instruction_edit_queue.py` | `image_unknown`, `image_binned` with `ids` |
| `/^\d+ batch/` | `more-seeds.js` 293 and 515 | "N batches × C is I images; a press may queue M." | `test_api_more_seeds.py` | `more_seeds_cap` |
| `err.message.includes(name)` | `generation.js` 558, `refine-rules.js` 121 (`dropPassLora`) | "no LoRA named 'x' in ..."; the backend's mismatch sentence | `test_api_conditioning.py` (missing only) | `lora_missing`, `lora_mismatch` |
| `/mask [0-9a-f]{16} ... has just gone ... no stored mask ... the file for mask/` | `inpaint.js` 428 (`maskGoneRefusal`) | `storage.mask_path`, `jobs.MaskGone` | `test_api_inpaint.py` | `mask_missing` |
| `/selection changed/i` | `gallery-actions.js` 985 | Delete unstarred's 409 | `smoke-delete` | `selection_changed` |
| `/intent\|receipt\|more seeds request/i` on a 409 | `more-seeds.js` 551 | the receipt ledger's two 409s | no | `more_seeds_receipt` |

The claims held: the tests of a sentence number well over twenty, and no
test would have failed on a rewording. The finding named four sentences;
the list found six more of the same kind (the outpaint presets, a
submission's sources and references, the edit's source, the mask, the
stale count and the receipt), which were built with them.

`backend/refusal_codes.py` holds the eight codes and `backend/refusals.py`
the refusal that answers `code` (and `ids`) beside an unchanged `detail`;
`api()` puts the code on the error and the page reads it through
`imageGone`, `imageBinned`, `refusedIds` and `loraRefused` (`state.js`).
No script in `web/` tests a refusal's message any longer. The cuts stand:
no response models were added and there is no OpenAPI snapshot. Two
answers became more exact on the way: an instruction edit's source is told
deleted from binned by the server, where the page guessed from its own copy
of the record, and a mask upload over a deleted source is said in the
dialog's words, where the capitalised sentence had not matched the page's
test. Left as it is: a pass dialog marking its Strength box where the
sentence it is held with mentions strength (`/strength/i`), which marks a
field and forgets, drops or resends nothing. Tests: `tests/test_refusal_codes.py` (each code beside its
sentence, the page reading exactly the server's codes, no message test in
`web/`); the rule is in `docs/rules.md` under Frontend and CLI and the
reasons in `docs/design-notes.md` under "Refusal codes the page reads".
The review of the build found two submission refusals still going out
without their code, both fixed in `submission.unusable_input`: a gone
source found by the upscaler's own look at it (`check_upscaler_source`),
and a mask swept between the submission's two looks at it.

## Tooling and the page

### T1 the smoke driver hears page errors

`DEV#38`. `tools/browser.mjs` never listens for `Runtime.exceptionThrown`,
so an uncaught page error fails nothing unless an `expect` depends on it.
In order: the driver records and then fails on them (report-only for one
`--all` first); a small `error`/`unhandledrejection` handler in the page
writing one deduplicated line to the error slot, the phone's only signal; a
no-undef pass in `tools/check_classic_scripts.mjs`. A Node test suite for
helpers was cut.

Built 2026-11-10. The claim held: the driver enabled `Runtime` and read
only the answers to its own calls. It now prints each
`Runtime.exceptionThrown` as `! page error:` with the step it arrived
during and exits 1 if any is left at the end; a rejection handled later
(`Runtime.exceptionRevoked`) is taken back, and what a step's own
expression throws stays that step's. The one whole run
(`run_smoke.py --all --jobs 2`, Windows, Edge, every page error printed
into the smoke's log) found none in 111 smokes, so the driver fails on them
with nothing to fix or allow first. `app.js` adds `sayPageError`
(`state.js`) for `error` and `unhandledrejection` before its other wiring:
one line, `Page error: <message> (<file>:<line>). Reload the page if
something has stopped working.`, in the panel's slot (over the bars on the
phone's Create view, in the gallery's slot on its Gallery view), written
once per distinct error. `tools/check_classic_scripts.mjs` has the
no-undef pass, over Node's bundled acorn (`--expose-internals`, passed by
the Python wrapper; no npm install), with the browser's globals a
hand-kept list; it found no undefined name in the 75 scripts. Tests:
`tests/test_browser_page_errors.py` (the real harness over a scripted
DevTools socket), `tests/test_classic_scripts.py`, and steps at the end of
`smoke-gallery-top` and `smoke-phone`, which dispatch the events by hand
because a real throw fails the run. The rules are in `docs/rules.md`
(Frontend, Tests), the reasons in `docs/design-notes.md` ("The smoke
driver hears page errors"). Not built: a step that allows a page error.
An error thrown while a script before `app.js` loads is not said on the
page (the driver hears it). `smoke-phone-gallery` fails on this PC at its
320 px "Search 160" fit, with or without this change; it is not a page
error.

### T2 the scripts in CI

**Local audit built 2026-11-11 (`DEV#39`).** Shellcheck 0.11.0 over all
eight Linux/WSL shell files initially found five diagnostics; blank-pattern
quoting and shell/source directives were fixed, and its final run is clean.
PSScriptAnalyzer 1.24.0 over the five PowerShell files reports zero errors
and 49 reviewed style diagnostics (46 Warning, 3 Information): console
`Write-Host` separation, helper names, positional helper calls, deliberate
manual dry-run handling and read-only probe catches. No correctness issue
was left from that review. Windows/Linux/WSL stub tests run without live
installation. The hosted installer-to-GPU-check job changes `.github`, so
that next step is USER, waiting for Matt's deliberate CI ruleset edit.
This PR changes no CI definition and claims no hosted clean install.

`DEV#39`. Ruff and Pyright are the only static checks; nothing runs
shellcheck, PSScriptAnalyzer or actionlint, and the installer has never run
on a clean machine (`DEV#28`). Run both linters locally once to see whether
they find anything real; then a hosted job running `install.ps1 -Yes` and
`linux/setup.sh` up to the GPU check. A workflow change lands only through
the deliberate ruleset edit. Coverage measurement was cut.

### T3 an injected accessibility audit

`DEV#40`. `phone_audit.js` checks geometry only. A `tools/a11y_audit.js` in
its shape (accessible names, duplicate ids, `aria-disabled` without a
reason, dialog labels) guards the held-button convention. It is a
regression guard and does not close `UI#7`. Shared step includes, a fast
tier and dynamic ports were cut.

Built 2026-11-10. The claim held: `phone_audit.js` reads boxes and type
sizes only. `tools/a11y_audit.js` defines `__a11yAudit` (unnamed controls,
duplicate ids, `aria-disabled` with neither a title nor described-by text
on the element, open dialogs with no label; natively disabled controls
listed, never failed), `__a11yWatch`, which audits every dialog as the
smoke's own steps open and shut it, and `__a11yCheck`, one `expect` that
prints what failed. 23 dialog smokes carry it: two steps after each
navigate and the check before the next and last. The first run found no
duplicate id, unnamed dialog or unnamed control, and four holds whose
reason was not on the held element, all fixed in `web/`: the Then refine
tick and More seeds' Refine each new image (the reason was the label's
title only), and the greyed Padding box of Only the masked area in the
Inpaint and Outpaint dialogs, which now says in its title that it is not
used while the tick is off. `smoke-phone-dialogs` opened the count question
without its label line, an empty label of the smoke's own making; its
message has the line now. Tests: `tests/test_a11y_audit.py` (the smokes'
shape, and the audit over markup of its own in a real browser, skipped only
without Edge or Chrome). The rules are in `docs/rules.md` (Frontend, Tests),
the reasons in `docs/design-notes.md` ("An injected accessibility audit").
Left: nothing found was left unfixed. Not covered: what a screen reader
says, focus, contrast, a glyph-only name (a ✕ counts as named), and dialog
states no smoke reaches; `UI#7` stands as it was.

### T4 the lightbox loads its neighbours

**Built and measured 2026-11-11 (`UI#41`).** A loading full PNG exposes
the current record's thumbnail underneath; failure leaves that thumbnail
with plain feedback. Request ownership and native event guards keep a late
PNG/load/error from replacing the next record or its feedback. After the
current PNG settles, at most one Image owner each way preloads neighbours;
close releases them. Existing zoom, compare and touch hit targets remain.
Cold loading/stale-event tests plus desktop lightbox and phone-gallery
smokes passed.

The [5,000-row receipt](../measurements/gallery-5000-2026-11-11.json) records a
Windows/Edge headless fake at 1400x900: exact 48 initial cards in 21.5 ms,
all 5,000 after 104 further pages in 19.082 s. Large jumps with pending
lazy thumbnails had long tasks; after 25 visible tail thumbnails decoded,
30 nearby 40 px steps measured 16.7 ms median / 16.8 ms maximum frame
interval. This fixture establishes no containment benefit, so card
containment remains unchanged. It is a working-tree CPU/browser receipt,
not a committed-tree, real-gallery, network, phone or card measurement.

`UI#41`. Each step sets the full PNG as the picture's source; nothing
preloads the next or previous, and no thumbnail stands in meanwhile. Preload
one each way after a step settles and show the cached thumbnail under the
PNG. Card containment for very large galleries is measured first (about
5,000 planted rows on a fake) before anything is built; a mid-size rendition
tier was cut.

### T5 phone Share without the settings

`UI#42`. `sharePhoneImage` shares the stored PNG, which carries the prompt,
settings, model, LoRA name and an inpaint's mask. The button's title says
so, but a touch screen shows no title. Share opens two rows, Picture only
(a canvas re-encode in the page) and With settings.

Built 2026-11-10. The finding held: `storage.save_image` writes the
`parameters`, `auradream` and mask text chunks, and Share sent that file
whole. Share on the phone's viewer now opens an action sheet
(`openViewerShare` in `web/phone-viewer.js`) with **Picture only** and
**With settings**. Picture only fetches the PNG, draws it on a canvas and
encodes it again (`pictureOnlyBlob`), so the file has the same pixels and
no text chunk; it is named `auradream-<id>.png`, since the stored file's
name holds the checkpoint and the seed. With settings is the stored PNG as
before. Where the browser shares no file (plain http to a LAN address) the
button still reads Download and its two rows download; a share refused for
want of the press downloads the same file it would have shared. A picture
that cannot be redrawn (over iOS's 16,777,216 canvas pixels, or a failed
encode) is an error in the viewer's line, never the stored PNG in its
place; a walk to another image or the next file that is made clears it,
as Download's failure is cleared. More's Download and the PC's lightbox
are unchanged.
`smoke-phone-gallery` checks the sheet, both files' chunks and pixels, the
two download paths and the two refusals in Chrome; a real phone's share sheet has not been
tried.

### T6 the guide from the page

**Built 2026-11-11 (`UI#43`).** Help links `/guide` in a new tab, preserving
the composer. The route reads only checkout-owned `docs/usage.md`, using
the already locked Markdown parser promoted to an explicit dependency:
tables, escaped raw HTML, stable duplicate-safe heading ids. Other repo
document links point to GitHub; no filesystem directory is mounted. Its
stylesheet sits outside the app's cascade. Route/render/security tests
verify local anchor targets, and the shortcuts smoke checks desktop and
narrow layouts. The passcode closes the new route like all other routes.

`UI#43`. Nothing in the page links `docs/usage.md`. There is no Markdown
library and no build step, so serving it needs a dependency or a small
renderer for its tables and anchors. First version: one read-only route,
a Guide link in the shortcuts dialog's head, and a test that every linked
anchor exists. A "why?" on each held reason was cut.

## Not taken

Checked and dropped as not worth it for one user, or settled before:

- A queue that survives a restart: the update and control scripts refuse a
  busy server, and Finish run covers grids.
- A schema version stamp: the per-connection check is a deliberate fix
  (2026-10-12), and additive columns make an older build harmless.
- Gallery indexes and per-row stats: a risk without a symptom.
- Tighter dependency bounds, an audit step: `uv.lock` already pins.
- Compressed or cached page delivery, a hidden-tab status poll: not taken
  before, in the design notes.
- A light theme; an i18n seam; a container image and a sandboxed unit.
- Colour matching for composites: the measured seam equals the VAE round
  trip.
- Saved quantized weights and 4-bit on Windows: the load's RAM peak was
  fixed 2026-10-25.
- A few-step Qwen-Image distillation and CFG rescale; LoRA stacking, LoKr
  and trigger words; a checkpoint grid axis and prompt lists; ComfyUI and
  JPEG metadata; keep rates beside the settings; machine-to-machine bundles;
  a command palette; a replayable lineage chain.
- Decided before: restructuring the routers and patched globals
  ([source audit](source-audit.md)), moving user data out of the checkout
  ([first run](review-2026-11-01-first-run.md)), a second curation mark
  ([validation](../design/validation.md)).
- Already rows: detection-driven masks (`INP#12`), other step caches
  (`PERF#4`), tiny-autoencoder previews (`GEN#6`), fit predicted from
  recorded peaks (`GEN#15`).
