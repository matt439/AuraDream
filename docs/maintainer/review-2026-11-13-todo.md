# TODO audit, 2026-11-13

Matt asked for an audit when the backlog had no `OPEN` row. This audit
writes owners and rows only: nothing below is built, no GPU run, model
download or real gallery operation was performed, and no decision was taken
for Matt. Each section here is the owner of the row it names until a plan
replaces it.

## Reconciliation

The starting index had 33 rows: no `OPEN` or `DECISION` row; the rest
`CARD`, `USER`, `BLOCKED` or `DEFERRED`. Every local link and anchor
resolved and every row was under 500 bytes. Each row was read against its
owner, the source and git history since 2026-11-09 (through PR #73). Fixed
in this commit:

- Review coverage: a blank line split the table before the 2026-11-12 row,
  and the `---` straight after that row made it a setext heading on GitHub.
  The row also had a stray `).;`, and said nothing of `GEN#46`, closed
  2026-11-13 by PR #73. NOW's ready-backlog sentence now names `GEN#46`
  beside `DEV#58`.
- The 2026-11-10 first-time row said F27 "is now under `GEN#65`", and its
  owner still called F27 `GEN#46`'s shard-map step; `GEN#65` recorded both
  FLUX 3 sizes 2026-11-12. Both now say so.
- `GEN#15`: the row's scope (a model-offload text-to-image render; the
  2026-11-11 sequential inpaint does not count) was only in the 11-12
  audit's answers. [plan-peak-vram.md](plan-peak-vram.md)'s phase 3 now
  records it.
- `INP#12`'s link now reaches the plan's desk results, not its top.

Matching their owners and left alone: `DEV#23`, `CARD#13`, the 3070 rows,
`INP#11`, `GEN#16`, `GEN#48`, the `PERF` rows, `GEN#6`, `GEN#66`, `GEN#68`
and `INP#12`'s status.

## Method

Two rounds. In the first, one agent reconciled the index; six proposed
work through separate lenses (composing, grids and the queue; the gallery
and reruns; models and the loader; defects in code landed since
2026-11-10; tooling, tests and docs; the page), each followed by a
skeptic told to refute every proposal (already built, decided against or
deferred, wrong evidence, a failure that does not trace, a Never-rated
feature, or card work), and a critic reviewed the lot. The critic named six
areas no lens had covered, and a second round ran over them: the Windows
installer and launchers, the CLIs, the prompt analyser, backups and
retention, the phone hosts, and WSL and native Linux.

Of 27 proposals, 4 were refuted (see Not added). The Windows sidecar race
was found by five lenses and the reconciliation, and is one row here. The
skeptics lowered several severities and corrected citations; the rows below
carry the corrected versions. The full suite passed on main (85691d4,
Windows): 9,704 passed, 254 skipped, none failed.

## Ready work (OPEN)

### DEV#64 Passcode renewal ignores the poll mark

`DEV#56` (af399cf) renews a day-old session cookie on a request
`counts_as_activity` counts. `GEN#67` landed later the same day and added
the `X-AuraDream-Poll` mark through a `polled` keyword that defaults to
False (`backend/keep_awake.py:114`). The passcode layer still calls
`counts_as_activity(method, path)` without it (`backend/passcode.py:411`),
so a marked poll renews. That breaks the module's own rule
(`passcode.py:78-82`, `:346-348`: never on a poll). The marked timers keep
firing in an idle tab: the catalogue retry while Models is open
(`web/model-switch.js:882-886`), the model-shots retry
(`web/model-shots.js:118-122`), the status previews after an idle release
(`web/status.js:783-787`) and the maintenance re-reads
(`model-switch.js:1167`, `web/model-storage.js:150`). Reproduced with the
fixtures of `tests/test_api_passcode.py`: after 24 h, a marked
`GET /api/models` answered with a fresh 30-day cookie, and a marked
`POST /api/grid/preview` did the same on its 422. The renewal test
(`:367-395`) never sends the mark. First step: call
`keep_awake.scope_counts_as_activity(scope)` instead, and add marked and
unmarked cases to that test. MED, size S.

### GAL#28 Windows sidecar reads and saves collide with os.replace

Four sidecars read with a plain `open()` and no lock, and publish with
`os.replace` under the gallery lock: `model_favourites.py:50`/`:95`
(read at `:104-106`, "Atomic replacement makes a read coherent"),
`snippets.py:37`/`:80`, `presets.py:40`/`:83` and
`saved_searches.py:41`/`:86`. On Windows a handle without
`FILE_SHARE_DELETE` makes the replace fail, and an open during the replace
fails too, so both directions raise `PermissionError`, which each module
turns into its "could not be read" (pointing the user at a file that is not
damaged) or "could not be saved" refusal. The routes are sync handlers, so
two tabs or devices can overlap. Probes on main, a writer thread toggling
a star against 20,000 reads: 40-209 reads and 55-263 saves refused across
runs (the 11-12 audit, which found the read half under `DEV#58` and left it
unfixed, saw 78 reads). A snippet probe refused 168 of 200 saves against a
tight-loop reader. Nothing is lost (the previous set is kept); the
visible cost is a rare refused star or save, and smoke-model-hover's last
step flaking on local Windows runs. `storage._rename_with_retry`
(`storage.py:826-841`) and `bundles.py:123` already handle this sharing
violation elsewhere. `startup_preferences.py`, `backup_settings.py` and
`backup_receipts.py` share the pattern but are read at startup or by the
backup service, not per request; the fix says whether its helper covers
them. First step: one shared bounded retry of `PermissionError` (winerror 5
or 32) for the read's open and the replace, keeping each module's words and
taking no lock on reads, with a Windows-only race test per sidecar (CI's
Windows pytest shards run it). LOW, size S-M.

### GAL#29 A deleted PNG a reader held stays and travels in bundles

Every permanent delete commits the row, then calls `storage._unlink` once
(`storage.py:1671-1677`), which returns False on any `OSError`; Delete
permanently, Empty bin, purge and delete_matching ignore the result
(`gallery_delete.py:279-289`, `:607-612`, `:549-554`, `:263-268`), as does
`_delete_thumbnails` (`storage.py:93-95`). On Windows a read handle without
`FILE_SHARE_DELETE` (the `/outputs` mount streaming the full PNG to a
lightbox or phone, or Defender holding a just-saved file,
design-notes.md:3971-3973) makes the unlink fail, and the file stays after
the handle closes (reproduced with a scratch probe). The comment at
`gallery_delete.py:283-286` says `clear_history`'s orphan sweep collects
it, but only the terminal `clear` (L4, Never) calls that sweep. A
whole-gallery export's `_inventory` (`bundles.py:234-268`; automatic backups
through `:620`) archives every PNG under outputs, row or not, and an import
installs it again (`:1154-1197`). So the deleted image rides every bundle;
the only sign is the Bundles dialog's records-versus-images line
(`web/bundles.js:397`). J8, J12 and J14 are Often. First step: retry the
unlink as `_rename_with_retry` does; record any file still held in a small
`pending_unlinks` table (a `_migrate` entry), swept at startup, at each
permanent delete and in `bundles.preview()` under the reservation. Never
glob-sweep untracked PNGs (`gallery_delete.py:226-227` refuses that for a
reason). Correct the comment. CPU tests hold a handle across `delete_image`
and `empty_bin`. LOW, size S-M.

### GAL#30 The Runs list opens one database session per run

The page asks for `GET /api/runs?limit=200` (`web/runs.js:478-480`) after
every finished job (`web/job-results.js:236`, `web/job-stream.js:390`) and
every bin, restore and bundle action. `gallery_reports.runs`
(`backend/gallery_reports.py:529-644`) calls `run_sizes` per listed run
(`:569`, defined `:511-526`), and each goes through
`run_jobs.retained_run_jobs` (`backend/run_jobs.py:78-93`): its own
`storage._session` (with `_migrate` and the gallery lock), a gunzip and a
`GenerateRequest` validation of every cell, only for distinct sizes. On a
scratch database of 200 runs of 40 cells, `runs(200)` took 1.4-1.6 s and
opened 202 sessions (up to about 400 when retained rows are missing). The
route runs in a thread, so the cost is a late picker and CPU after each job
(E4, A13, D3 Often). First step: read every listed run's retained cells in
one session, taking sizes from the raw request JSON without validation (a
malformed row still falls back to the images table); keep `run_sizes` for
single-run routes; test the session count. LOW, size S.

### GEN#69 Grid refusals name schema fields

Reproduced on a fake, `POST /api/grid/preview`: a variable block over 2,000
characters answers "prompt_axis: String should have at most 2000
characters" (`schemas_generate.py:760`; shown by `web/state.js:838-847`),
naming the field rather than the Prompt variables box, which has no limit
or counter (`web/index.html:762-764`) and holds half the prompt's 4,000.
And `nan` in the CFG or Mu axis passes `grid.parse_axis`
(`backend/grid.py:255-282`) and `_within` (`:627-652`, whose docstring says
it exists to prevent exactly this), so pydantic answers "guidance_scale:
Input should be less than or equal to 20". `inf` and a `nan` step are
already refused in words. C7 is Often; the length case is the realistic
one. First step: refuse a non-finite axis value as not a number; give the
block's length a worded refusal under the box's name with its count and
limit (2,000 kept, with `PresetGrid.prompt_axis`, unless Matt wants 4,000).
Grid and API tests. LOW, size S.

### GEN#70 Models' Speed facts miss the warm fp8 repeats

The picker's Speed fact reads only `model_benchmarks._SAMPLES`
(`backend/model_benchmarks.py:61-157`). The 10-12 review's R15 asked for warm
repeats of the four "First run, busy PC" samples
(`review-2026-10-12.md:362-375`); the card recorded them on 2026-10-25 at the
samples' own defaults (fp8, model offload, Windows, 1024x1024,
[verified.md](../verified.md) lines 578-624): Animagine 7.60 s, Illustrious
7.61 s, Z-Image Turbo 10.17 s, FLUX.2 klein base 4B 57.95 s, ERNIE-Image
Turbo 13.14 s. No commit took them in. On main, Animagine (the open
checkpoint a first visit suggests) shows "Reference, ~17 s, First run, busy
PC"; klein base shows "Unmeasured"; ERNIE's only sample is the 4-bit Linux
one. The labels are honest, so this is stale data rather than a false
display. K2 is Often. First step: add warm fp8/model/win32 samples for the
five with verified.md's caveat (other apps running), keep the older ones as
fallbacks, leave out int8, bf16 and sequential (K4, K5 Never), and decide
in the same change whether the table's other fp8/model repeats (klein 4B
4.70 s against 10.2 s, Kohaku 7.85 s against 9.68 s) replace theirs. Extend
`tests/test_model_benchmarks.py`; rerun smoke-model-hover. LOW, size S.

### GEN#71 Models > On disk shows an unrecorded revision as 0 bytes

`model_cache_storage` keys ownership by the recorded commit (`:96-111`) and
matches `revision.commit_hash` only (`:135-136`); a catalogue repository
cached at any other commit goes to the unreferenced bytes (`:141-143`), and
the model row sums to 0, not None (`:169-170`). The page then drops the row
(`web/model-storage.js:49`), may say "No catalogue checkpoint has files in
the cache" (`:52`) and lists the repository under Other repositories
(`:88-89`). Models' own cache column calls the same case Unknown on
purpose (`backend/model_info.py:403-411`, "main is mutable"), and the
design notes promise unknown references reported honestly
(design-notes.md:15035-15038). Reproduced with `snapshot` over a scratch
Hub layout. Any download made before a recorded commit, or after a later
upstream commit, lands here, and `GEN#68`'s guarded delete would be
designed on this classification. First step: mark a revision of a
catalogue repository at another commit as that checkpoint's other
revision, with `cached_bytes` None and its bytes shown under the
checkpoint's name; keep an old revision beside the recorded one
unreferenced. Add the case to `tests/test_model_cache_storage.py` and a
smoke-model-storage step. Best before Matt's `GEN#68` look. LOW, size S.

### PA#17 Tag verdicts miss Danbooru aliases

The chip and Swap tags verdict asks Danbooru for exact names only
(`search[name_normalize]`, `backend/tags.py:353`). Danbooru keeps an alias's
antecedent as an empty tag: on 2026-11-13 `twin_tails` and `blond_hair`
came back with post_count 0, while the active alias table maps them to
`twintails` and `blonde_hair`. So `_validation` answers "Danbooru tag has no
posts" (`tags.py:418-419`), the chip shows a red ? (`web/tag-chips.js:209-215`)
without naming the canonical tag it can show for valid tags (`:190-195`),
Check leaves the item unplaced as having "no usable Danbooru category"
(`backend/prompt_check.py:126-127`, via `tag_lookup.py:175`), and Swap tags
(`web/tag-swap.js:419`) and Refine (`web/refine-prompt.js:474`) get the same
verdict. Autocomplete already resolves aliases (`tags.py:247`). This is
Danbooru's own redirect, not the "did you mean" the 10-04 review declined
(U113, `review-2026-10-04-ui.md:4225`). B3, B5, B7 and G8 are Often. First
step: for names missing or empty, one chunked active `/tag_aliases.json`
lookup, then the consequents' rows, both cached; a new `alias` code with the
canonical set; Check takes the consequent's category; the chip names it.
Stub-fetch tests in `tests/test_tags.py` (alias, chained miss, lookup failing
back to `empty`) and a smoke-tags fixture. LOW, size S.

### UI#54 The offline page speaks every 10 s; held buttons give no reason

`web/offline.html:64` makes `#state` a polite live region, and each probe
writes "Trying…" then "No answer at <time>…" (`:83`, `:96`) every 10 s
(`:71`, `:102`), so a screen reader hears a new pair for as long as the PC
sleeps (observed on a fake: four announcements in 21 s). The main page keeps
its clock out of what is announced on purpose (`web/job-progress.js:5-10`).
Its held Try again (`offline.html:82`) and passcode's held Sign in
(`web/passcode.html:102`) are `aria-disabled` with no title, against the
Frontend rule; `tools/a11y_audit.js` flags Try again. Neither smoke runs the
audit (`tests/test_a11y_audit.py:29-36`). `sw.js` caches offline.html only
at install (`:17`, `:27`) and rules.md requires a version bump on a change,
which `tests/test_service_worker.py:39` does not enforce. D10 is Often.
First step: announce only changes of outcome, with the clock outside the
live part; titles for both held buttons; bump the cache version and pin
offline.html's hash to it in a test; the audit steps in smoke-offline and
smoke-passcode, both added to `WATCHED`. Hands-on screen-reader checks stay
`UI#7`. LOW, size S.

### UI#55 Notify when done on a phone through the service worker

The phone moves Notify into More options (`web/phone.js:39-41`) and the
HTTPS recipe names notifications among what it gives a phone
(`docs/setup.md:379`), but the page's one delivery path is
`new window.Notification(...)` (`web/job-progress.js:548-551`). Chrome on
Android exposes the constructor and lets permission be granted, then
throws "Illegal constructor" on use, so the first hidden finish lands in the
catch (`:552-556`): Notify unticks itself and says notifications could not be
shown. A service worker is registered on secure pages
(`web/app.js:971-974`), yet `showNotification` and `notificationclick`
appear nowhere in `web/`. Traced from the code and Chromium's documented
behaviour, not reproduced: emulation keeps the constructor, and
smoke-jobs' stub (`:462`, `:509`) is constructible. Notify is unrated. First
step: with a controller, `(await navigator.serviceWorker.ready).showNotification`
first, the constructor as fallback; a `notificationclick` handler in `sw.js`
that focuses or opens `/`; a smoke-jobs step whose constructor throws and
whose registration records the job's tag. Real Android and iOS delivery
joins `UI#40`'s list; a suspended phone (Web Push) stays out of scope. LOW,
size S.

### DEV#65 smoke-compact-setup's reorder races the loras refresh

After the smoke's FLUX.2 klein switch (`tools/smoke-compact-setup.json:866`),
a switch resets `lorasLoaded` (`web/model-switch.js:1290-1291`) and the next
status fires an unawaited `refreshLoras()` (`web/status.js:767-770`); its
answer runs `syncConditioning()` (`:944`) and `renderReferences()`
(`web/sizes.js:511`), which empties the chip list (`:523`). If that lands
between the focus (`:884`) and Alt+ArrowRight (`:887-891`), the focused x is
gone and the expect at `:894` fails. Seen once in four local runs (in a
scratch copy with the audit steps added); the cause was reproduced directly
by calling `syncConditioning()` with a chip focused. I4 is Never, so the fix
is the smoke's. First step: wait for the post-switch loras answer before
focusing, proved with a scratch copy that delays `/api/loras`. LOW, size S.

### DEV#66 linux/update.sh never checks the installed unit

`DEV#55` (c8533b0, 2026-11-12) added `RestartPreventExitStatus=78` to
`install-service.sh`'s unit (`:204-206`); any unit installed earlier lacks
it, so a refused passcode or TLS setting restarts every 10 s up to the
start limit (`:191-192`, `:203`). The 11-12 results say a service gets it
only when install-service.sh runs again. `linux/update.sh`, the documented
update path (setup.md:825), uses the unit only by name (`:53`, `:114`) and
never compares it with `unit_text` (`install-service.sh:185`), so this and
any later unit change drift silently. Whether the real server's unit lacks
the line is inferred, not checked. First step: after the pull, compare the
installed unit with what install-service.sh would write and say so in one
line naming the command when they differ; sandbox tests for a stale and a
current unit. LOW, size S.

### DEV#67 bin\auradream.ps1 lost step with its .cmd

c2de5f0 (`DEV#51`) gave `bin/auradream.cmd:6-18` a uv lookup (PATH, then
`~\.local\bin`, then WinGet's Links) and an "uv was not found. Run
install.cmd" line with exit 1. `bin/auradream.ps1`, which PowerShell picks
first, still runs a bare `& uv run ...` and exits `$rc`, never set when uv
is missing (`:7-12`): reproduced with PATH cut to System32, it printed
PowerShell's CommandNotFoundException and exited 0. docs/rules.md:1126-1130
keeps each pair in step. `update.ps1:155-160` looks only at `Get-Command`
too. First step: the .cmd's lookup, line and exit 1 in the .ps1 (a null
exit code is 1), with a Windows test in `tests/test_install_scripts.py`;
optionally the same fallback in update.ps1. LOW, size S.

### DEV#68 update.ps1 checks a relative gallery path from the wrong folder

`update.ps1:170-171` only expands `AURADREAM_DB_PATH`, and `:177` tests it
against PowerShell's current folder; `update.cmd:13` does not change
folder, and the script's only `Push-Location` (`:281`) comes after the
copy. Everything else reads a relative path from the project folder (the
bin wrappers push there; `config.py:107`; `install.ps1:303`;
`linux/update.sh:156-158`). Reproduced on a scratch checkout: run from
another folder, the dry run says there is no gallery to copy; from the
project folder it names the copy. A real run would then pull and sync with
no copy to go back to (setup.md:520-541). First step: join a relative path
to `$Root` before the test; a test with the sandbox's cwd elsewhere. LOW,
size S.

### DEV#69 update.ps1 on a detached HEAD: noise and no named stop

`Invoke-Capture` (`update.ps1:49`, `install.ps1:124`) stringifies `2>&1`
records with `"$_"`, so under PowerShell 5.1 a blank stderr line prints as
`System.Management.Automation.RemoteException`. setup.md's "Going back
after an update" (`:546-569`) has the user `git checkout <commit>`, which
detaches HEAD; the next update.cmd then prints Git's "You are not currently
on a branch.", two RemoteException lines and the generic stop, since no
pattern at `:249-258` matches, though `:255-256` already offers
`git checkout main` for another case. Reproduced on a scratch clone. First
step: map an ErrorRecord to its message (blank stays blank) in both scripts;
a detached-HEAD branch naming `git checkout main`; extend the `_git` stub;
check `linux/update.sh` for the same case. LOW, size S.

### DEV#70 wsl/env.sh keeps a stale default when sourced again

`wsl/env.sh:47-54` recomputes `AURADREAM_MODEL_ID` only when it is empty,
so a value still equal to its own marker survives a change of
`AURADREAM_QUANT` or `AURADREAM_NUNCHAKU_FORMAT`, against its comments
(`:41-46`, `:60-62`). setup.sh recommends sourcing it from `.bashrc` and
`serve.sh:8` sources it again, so the second source is the normal path.
Reproduced with a scratch HOME: after `export AURADREAM_QUANT=fp8` the
lite nvfp4 id stays. The marked id is not explicit
(`startup_preferences.py:226-228`), so with no saved choice or under
`--defaults` the fp8 server takes the lite repo and `refuse_prequantized`
(`loading.py:528-546`) tells the user to change a variable they never set.
With a saved choice nothing goes wrong. No test runs env.sh. First step: a
marked value is treated as unset (recomputed under nunchaku, dropped with
its marker otherwise), the same for the quantization marker; a bash-gated
`tests/test_wsl_env.py` sourcing twice in four cases. LOW, size S.

## Matt's calls (DECISION)

Each has a recommended answer, which is not taken until Matt says so.

- **`GAL#31` free words in Delete unstarred.** Delete unstarred reads the
  search through `search.strict_scope` (`backend/api_gallery.py:31-45`,
  `backend/search.py:393-414`), which refuses any free word, so under a
  search like `fox` the button shows, the gallery lists the matches and the
  press is refused. That is a recorded rule (design-notes.md:7302-7306,
  smoke-pinned), and on 2026-10-29 Matt kept refusing `-word` for the same
  reason (`plan-search-exclusion.md:139-157`): a destructive filter is not
  resolved through the full-text index. Since `GAL#25` each free word is a
  prefix match (`storage.py:930-936`), so `fox` also takes `foxglove` and
  `foxes`. The safeguards are the typed and pinned count, the sample
  prompts in the dialog, and that it only bins. J6 and E5 are Often.
  *Recommended:* allow positive free words in the page's Delete unstarred
  only, through the same prefix match, refusing a search that would take
  the odd-quote fallback and still refusing `-word`; `clear --where` stays
  strict. Size S.
- **`UI#56` a key for Reuse.** G1 (Reuse settings) and L6 (shortcuts) are
  Often, but Reuse has no key: the lightbox takes F, R, S, X and Delete
  (`web/app.js:817-823`), the workspace /, P and M (`:800-805`), and help
  lists no Reuse key (`web/shortcuts.js:33-40`). Today it is the context
  menu or a Tab to `#lb-reuse`. The keyboard plan kept letters few on
  purpose (`plan-shortcut-workflow.md:123-129`, `:200-202`). *Recommended:*
  U only, in the lightbox (pressing its Reuse button, so its hold and title
  apply) and on a focused card or sheet cell scoped like M, with help rows,
  `aria-keyshortcuts` and Reuse's existing Undo; smoke-shortcut-dispatch
  extended. Size S.

## Matt's answers (2026-11-13)

Matt accepted both recommendations the same day: `GAL#31` (positive free
words in the page's Delete unstarred, prefix-matched as the gallery
matches them, refusing the odd-quote fallback and `-word`; `clear --where`
stays strict) and `UI#56` (U for Reuse settings in the lightbox and on a
focused card or sheet cell, scoped like M). Both are `OPEN`, with the first
steps above. `GAL#31` changes a recorded rule: its build updates
design-notes.md's Delete unstarred section (7302-7306) and the smoke step
that pins the free-text refusal.

## Results

- **`DEV#64`** (done 2026-11-13). `PasscodeMiddleware` asks
  `keep_awake.scope_counts_as_activity(scope)`, the rule `ActivityMiddleware`
  uses, so the poll mark is read. `test_a_request_the_page_marks_as_a_poll_never_renews`
  sends a marked `GET /api/models` and a marked grid preview to a day-old
  session (no cookie), a marked unload (renewed: not a polled route), and
  both requests unmarked (renewed); it fails on the old call. The design
  notes' renewal paragraph and rules.md name the rule. Passcode, keep-awake
  and docs tests, ruff, the type gate, smoke-passcode and
  smoke-idle-activity pass.
- **`GAL#28`** (done 2026-11-13). `backend/sidecar_io.py` is the one
  `open_read` and `replace` of all seven sidecars: the four the row named
  and startup choices, backup settings and receipts. Both retry for about
  half a second (six waits, 10-200 ms) on Windows only, and only on a
  sharing violation. One thing the probes had not shown: `os.replace`
  reports it as WinError 5 or 32, but a refused `open` goes through the C
  runtime and arrives as EACCES with no `winerror`, so that is the reader's
  case. Any other error, and every error off Windows, is raised at once, so
  each module's own refusal words stand. `tests/test_sidecar_io.py`: the
  retry and the give-up on both sides, the errors not retried, a scan that
  every sidecar goes through the helper, and on Windows a three-second
  star-against-read race that collected 61 refusals before the fix and none
  after (three runs). The 1,065 tests of the sidecars and their routes, ruff,
  the type gate and four runs of smoke-model-hover pass.
  Follow-up, the same day: the retry alone was not enough. Under a reader in
  a tight loop, about one run in five still had a save refused after its
  half second. The four request sidecars' reads now also take
  `sidecar_io.reserved_read`: `gallery_lock.local`, this process's part of
  the sidecar's reservation, which a save holds across its replace. It never
  takes or creates the lock file, so the recorded rule that a read never
  waits on another process's save still holds (the cross-process tests of
  setups and saved searches). The race then passed 12 runs of 12 alone. Under
  parallel load its only failure was the test's own floor of 100 reads (96),
  now 10.
- **`GAL#29`** (done 2026-11-13). `backend/deleted_files.py` removes a
  permanently deleted image's PNG, thumbnail and legacy sibling after its
  row. Each file is retried as `sidecar_io` does. One still held is
  remembered for the process with the image filename it belonged to, and
  swept after every permanent delete, before every bundle `_inventory` and
  at exit. Delete permanently, Empty bin, purge, the scoped delete and
  `clear` all use it. A sweep never removes a file a row names again or any
  other untracked file. An independent review found that the
  sweep in `_inventory` broke Bundle & clear: a file held at the first
  inventory and released before the check after writing was listed once,
  swept the second time, and the clear was refused as an outside change
  with the deleted image in the archive. `_inventory` now leaves out a file
  still remembered, so an operation's inventories agree. Following the same
  review, a sweep never raises (the delete before it has happened), and the
  exit sweep waits half a second for the gallery, never behind a running
  bundle. `tests/conftest.py` resets the memory for every test.
  `tests/test_deleted_files.py` (10 tests, one with a real open handle on
  Windows): five failed without the change; both Bundle & clear cases
  (held throughout, released between the inventories) failed without the
  exclusion. The full suite passed twice (9,732), ruff and the type gate
  pass, as do nine smokes of the sidecars, delete, bin and bundles, twice in
  parallel. Not explained: one earlier full run had 3 failures and an error
  that the saved output did not name, and one parallel run of
  smoke-bin-restore-order failed and then passed seven times. Its fake's log
  shows no error.
- **`GEN#71`** (done 2026-11-13). `model_cache_storage._snapshot` notes each
  repository's recorded commits. A catalogue model repository with no
  cached revision at any of them is the other revision of every checkpoint
  sourcing it (`other_revision_of` on the revision): its bytes are those
  checkpoints' `other_revision_bytes`, their `cached_bytes` is null, and
  they are left out of the unreferenced total. An old revision beside a
  recorded one stays unreferenced, as before. The page shows such a row as
  "Unknown (size)", titled with why, and keeps the repository out of Other
  repositories. `tests/test_model_cache_storage.py` adds the case and holds
  the old revision; the fake plants Kohaku XL Zeta at an unrecorded commit
  and smoke-model-storage checks its row (the phone fit included). The
  storage, model-info, bundle and fake-server tests and ruff pass.
- **`GAL#30`** (done 2026-11-13). `gallery_reports.runs` reads every listed
  run's sizes in its own session (`_listed_run_sizes`): one query of the
  listed runs' retained cells, sizes taken from the raw request JSON
  without a `GenerateRequest` per cell, then one images query for the runs
  with an unreadable retained row or none. `run_sizes` had no other caller
  (Finish run works out its own sizes) and is gone.
  `tests/test_runs_list_sizes.py`: 30 runs list in two sessions (the listed
  runs' and the recovery candidates') with nothing validated, where the old
  code opened 32; an unreadable row falls back to its images without
  touching another run's sizes; a run with no retained row takes its
  images'. The Runs, storage, size-crop, Finish run and persistence tests,
  ruff, the type gate and smoke-runs-scope-recovery,
  smoke-runs-favorite-counts and smoke-size-crop pass.
- **`GEN#69`** (done 2026-11-13). `grid.parse_axis` refuses a value that is
  no finite number (`nan`, `inf`) as "CFG: 'nan' is not a number", under the
  box's name like any other bad token. `GridRequest.prompt_axis` has no
  field bound now: `_prompt_axis_fits`, a model check whose refusal carries
  no field name, says "Prompt variables: N characters, more than the 2,000
  they hold" with the block's count. The limit stays 2,000 (`PROMPT_AXIS_LIMIT`), as a saved
  setup's `PresetGrid.prompt_axis`, which a test holds equal. Grid and API
  tests (preview and run, both cases, the limit itself accepted) fail
  without the change. The full suite (9,745), ruff, the type gate,
  smoke-grid and smoke-random-prompts pass.
- **`GEN#70`** (done 2026-11-13). `model_benchmarks._SAMPLES` takes the
  2026-10-25 table's identical repeats at fp8, model offload, Windows, the
  profile's steps and CFG: Animagine 7.60 s, Illustrious 7.61 s and Z-Image
  Turbo 10.17 s (Fast) replace their 2026-10-05 busy first runs, and klein
  base 4B (57.95 s, Slow) and ERNIE-Image Turbo's FP8 (13.14 s, Fast) get
  their first Windows samples. Each note names the process's first render
  and that other apps were running. The busy first runs were removed rather
  than kept as fallbacks: at the same setup a fallback is never chosen.
  Schnell keeps its busy first run, the 4-bit Linux samples stay, and int8,
  bf16 and sequential offload stay references (K4, K5). Decided: klein 4B's
  4.70 s and Kohaku's 7.85 s do not replace their 10.2 s and 9.68 s, which
  were already warm repeats with the same category. `test_model_benchmarks.py`
  holds the five (label, timing, steps, setup, the other modes unmatched)
  and the Linux samples beside them. Its 26 tests, smoke-model-hover,
  smoke-model-switch, smoke-model-startup, smoke-ernie and smoke-animagine
  pass.
- **`PA#17`** (done 2026-11-13). `DanbooruTagVocabulary.validate` asks the
  active alias table (`/tag_aliases.json`,
  `search[antecedent_name_comma]`, checked live the same day: `twin_tails`
  and `blond_hair` answer `twintails` and `blonde_hair`) about names with no
  row or an empty one, chunked as the tag query is, then the consequents'
  rows; both cached. `tags.alias_verdict` gives code `alias`: still not
  valid, the consequent as `canonical` with its category and post count,
  and a reason naming it, which the chip's label ("alias of twintails"),
  its tip, Swap tags' and Refine's titles show. Check places it by that
  category. A consequent that is no usable tag at the cutoff keeps the
  name's own verdict, and no chain is followed. A failed alias lookup
  caches nothing, leaving `empty` or `not_found`. `TagValidationItem.code`
  takes `alias`: the first smoke run found the response refused without
  it, which an API test now holds. The fake answers `twin tails` the same
  way through `alias_verdict`. Tests: `tests/test_tags.py` (alias, empty
  consequent, too-new consequent, failed lookup then retried, cache
  counts) and `tests/test_api_analysis_tags.py` (the route and Check's
  category); smoke-tags (the chip, its tip and its order step),
  smoke-analysis-tags, smoke-tag-finder, smoke-swap and smoke-refine-sdxl
  pass.
- **`UI#54`** (done 2026-11-13). `offline.html`'s live part is written only
  when its words change ("No answer yet. Trying again every few seconds…",
  then nothing more while the PC stays silent); each ask's clock ("Asking
  now…", "Last asked at …") is in a `#clock` outside it. Try again and the
  passcode page's Sign in carry a title while held. `sw.js`'s cache is
  `auradream-offline-v2`, and `tests/test_service_worker.py` records
  offline.html's hash under it (`OFFLINE_RELEASES`), so a change without a
  new version fails; another test holds the one live part. smoke-offline
  and smoke-passcode carry the a11y audit on every page they show (both in
  `WATCHED`), re-injected after each in-page document swap; smoke-offline
  audits Try again while held and watches the live part through a second
  failed ask (no change), and smoke-passcode audits Sign in held on a
  wrong passcode. Both smokes and the audit's file checks pass. Hands-on
  screen-reader checks stay `UI#7`.
- **`UI#55`** (done 2026-11-13). `notifyJobDone` shows its notification
  through `(await navigator.serviceWorker.ready).showNotification` when a
  worker controls the page; a refused registration falls back to the
  constructor, and only both refused untick Notify. With no controller (plain
  http to a LAN address) the constructor runs as before. `sw.js`'s
  `notificationclick` closes it and focuses an open AuraDream tab, or opens
  `/`. Not pinned: `offline.html` is unchanged, so the cache keeps its name.
  smoke-jobs stubs `navigator.serviceWorker` (no controller for the steps
  before) and adds three steps: the constructor throws while the
  registration records the job's title, body and tag, Notify stays ticked;
  the registration refuses and the constructor takes it; both refuse and
  Notify unticks with its reason. The service-worker test holds the handler.
  smoke-jobs and the service-worker tests pass. Delivery on a real Android
  phone and an iPhone joins `UI#40`.
- **`DEV#65`** (done 2026-11-13). smoke-compact-setup's FLUX.2 klein switch
  step now counts the `/api/loras` answers through its own `api` wrapper and
  waits for one (and `lorasLoaded`) before the chips are set and focused; the
  step's `why` names the race. Proved with scratch copies whose
  `/api/loras` is held until a reference chip has focus (4 s at most): the
  old smoke fails at the reorder's expect, the new one passes. The smoke
  passes as committed.
- **`DEV#66`** (done 2026-11-13). `install-service.sh --print-unit` prints
  only the unit it would write (its definitions now come before anything
  runs) and changes nothing. `update.sh`'s last step, dry runs included,
  compares an installed unit with that output from the version now in the
  checkout and, when they differ, names the unit and `bash
  linux/install-service.sh` in one note; it never rewrites the unit. A
  failed print says nothing. `tests/test_linux_scripts.py`: `--print-unit`
  equals the unit `--no-start` writes and touches nothing; a stale unit
  (without `RestartPreventExitStatus=78`) is named and a current one is not,
  in a real run and a dry run, the file unchanged. The stale case fails on
  the old `update.sh`. The Linux script, control and passcode tests (251)
  pass under WSL. Whether the real server's unit lacks the line is still
  unchecked: the next update says so.
- **`DEV#67`** (done 2026-11-13). `bin/auradream.ps1` finds uv as its
  `.cmd` does (PATH, `~\.local\bin`, WinGet's Links), says "uv was not
  found. Run install.cmd from the AuraDream folder, then try again." and
  exits 1 without it, and exits 1 when the run left no exit code.
  `tests/test_startup_refusals.py` runs the real launcher under Windows
  PowerShell with an inert uv in each place and none: arguments, folder and
  exit code pass through, and the missing case says so with no
  CommandNotFoundException. Six of its eight cases fail on the old
  launcher. `update.ps1`'s own uv lookup (the optional part) followed
  after `DEV#68`: the same two folders after PATH; a dry run with uv only
  in each passes, and with none still stops on "uv is not installed" (the
  two found cases fail on the old script).
- **`DEV#68`** (done 2026-11-13). `update.ps1` reads `AURADREAM_DB_PATH` as
  `config.py` does: environment variables expanded, a leading `~` the user's
  folder, and a relative path joined to the project folder before the
  test, so the copy no longer depends on the window's folder.
  `tests/test_install_scripts.py` runs a dry run from another folder (with
  an empty `data` decoy there) for `data\gallery.db` and
  `~\data\gallery.db`; both name the copy beside the real gallery and fail
  on the old script, which said there was nothing to copy. `_run` takes a
  `cwd`. The install-script tests pass.

## Not added

- **Load a grid run back into the panel.** Matt deferred it on 2026-10-18
  (`plan-continue-run.md`, Q13, `:737-739`, `:774-775`); it returns only when
  he raises it.
- **The CLIs' `GalleryBusy`.** `stats` (`stats.py:128-132`), `reindex`
  (`reindex.py:1159-1163`) and generate's `init_db` (`generate.py:1227`) let
  `GalleryBusy` escape as a traceback while another process holds the
  gallery, and `clear` returns 3, which means its delete ran, when
  `clear_history` refuses with it (`clear.py:337-341`, `:151-153`,
  `:217-219`). Real, but L1, L3, L4 and L5 are all Never.
- **`stats` with a mistyped run or job** says "Nothing to compare" and exits
  0 (`stats.py:136-138`). L3 is Never.
- **Refusing an NVFP4 repo on a pre-Blackwell card before download.**
  `GEN#42` keeps a chosen format and lite repo whatever the card, pinned by
  `test_a_chosen_format_is_kept_whatever_the_card`, and whether the kernels
  fail on sm_86 is the card's to say (`CARD#15`).

## Validation

See the commit. Links and anchors in the changed files were checked, and
the docs-map tests and ruff were run. No code changed.
