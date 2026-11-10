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
[reported use](design/preferences.md) rates Often: [R1](#r1-refine-on-the-families-that-have-none)
and [R2](#r2-tiled-refine).

## Matt's answers (2026-11-10)

- **A passcode for the LAN service: yes.** [S2](#s2-passcode-beyond-loopback) is ready work.
- **Which non-refining checkpoint to enlarge from: not sure.** [R1](#r1-refine-on-the-families-that-have-none) stays a decision.
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

First step: Matt names the checkpoint he most wants to enlarge from. Then
read that pipeline's `__call__` to see whether `latents` plus truncated
sigmas can express the pass without a copied pipeline.

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

### R6 a full disk pauses the job

`GEN#52`. `jobs._save` retries only on `GalleryBusy`. Any other error from
`storage.save_image` is a cell failure, and the job stops after three in a
row, so a full disk costs up to three renders. Saved images are kept and
Finish run resumes the rest. Fix: treat `ENOSPC` and SQLite's "database or
disk is full" like the busy branch, holding the render in memory and pausing
with a plain-words reason. A submit-time space check was cut.

First step: a CPU test making `save_image` raise each error, to pin today's
behaviour (the failure count, a leftover `.png.part`) before the pause.

### R7 prompt recall

`GEN#53`. A prompt is recalled only through its image (Copy prompt, Reuse,
search). First slice: Use prompt beside Use seed in the context menu,
writing prompt and negative through `rewriteBox`. Only if that falls short:
Recent prompts under Examples (no panel height), a model-free read of
distinct recent pairs. A gallery-derived list cannot hold prompts whose jobs
saved nothing.

### R8 a snippet target for grid variables

`GEN#54`. Snippets target Prompt or Negative only
(`schemas_snippets.py`); a variable block persists only inside a saved
setup. A third target appends a saved `name = a | b` block to the grid's
variables box as Append does today, leaving expansion and reproduction
untouched. Server-side resolution of a bare `{name}` and style templates
with a `{prompt}` slot were cut. Starts as an addendum to
[the snippets plan](plan-prompt-snippets.md).

### R9 storage view

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

### S3 an HTTPS route

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

### S5 a log file

`DEV#35`. Logging is `basicConfig` to stderr; job, cell, load and switch
failures log their traceback only at DEBUG, and uvicorn's level is
hard-coded. On Windows a failed overnight job leaves nothing once the
console closes. Fix: a rotating file beside the database that always
receives tracebacks, `faulthandler` into it, uvicorn following the
configured level; the console unchanged. The file holds prompts and paths,
so it is never bundled, exported or treated as gallery data, and is
installed in `run`, not at import.

### S6 the running commit

`DEV#36`. The version is a static `0.1.0` and there are no tags.
`auradream-server status` prints the checkout's commit, which differs from
the running process after a pull without a restart. Read the short commit
once at start-up (None when there is no git), log it, and add it to
`/api/status` and the doctor. An in-page Machine check was cut to optional.

### S7 error codes the page reads

`DEV#37`. Over twenty sites in `web/` test English server wording
(`/unknown image/`, `/recycle bin/`, `/^\d+ batch/`, the LoRA's name) in
`err.message` to decide whether to forget or bin an id. A reworded refusal
would silently stop that cleanup. Fix: a stable `code` beside `detail` on
the few refusals the page acts on, the page reading the code. Response
models on every route and an OpenAPI snapshot were cut.

First step: list each `err.message` test in `web/`, the server sentence it
depends on and whether a test already pins it.

## Tooling and the page

### T1 the smoke driver hears page errors

`DEV#38`. `tools/browser.mjs` never listens for `Runtime.exceptionThrown`,
so an uncaught page error fails nothing unless an `expect` depends on it.
In order: the driver records and then fails on them (report-only for one
`--all` first); a small `error`/`unhandledrejection` handler in the page
writing one deduplicated line to the error slot, the phone's only signal; a
no-undef pass in `tools/check_classic_scripts.mjs`. A Node test suite for
helpers was cut.

### T2 the scripts in CI

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

### T4 the lightbox loads its neighbours

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

### T6 the guide from the page

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
  ([validation](design/validation.md)).
- Already rows: detection-driven masks (`INP#12`), other step caches
  (`PERF#4`), tiny-autoencoder previews (`GEN#6`), fit predicted from
  recorded peaks (`GEN#15`).
