# TODO audit, 2026-11-12

Matt asked for an audit of the backlog and, if there was too little work, a
brainstorm of what could be added. This audit writes owners and rows only:
nothing below is built, no GPU run, model download or real gallery operation
was performed, and no decision was taken for Matt. Each section here is the
owner of the row it names until a plan replaces it.

## Reconciliation

The starting index had 44 rows: one `OPEN` (`GEN#48`), 23 `CARD`, six
`USER`, 11 `DEFERRED`, three `BLOCKED` and no `DECISION`. Every local link and anchor resolved. Fixed in this commit:

- NOW still said no merge was requested; PR #69 merged on 2026-11-11. The
  Decisions paragraph was a running history. It now keeps only the
  decisions that still limit work, and the history stays in the owners.
- Review coverage for 2026-11-10 peers left out `GEN#50`, `GEN#55`'s
  first slice, `DEV#33`, `UI#41` and `UI#43`, which were built on 2026-11-11.
  The first-time row now says all 27 of its rows are built.
- `DEV#20`: stage 2 was in effect done in the 2026-11-11 Animagine fp8
  session (`CARD#13`). [Card guard](plan-card-harness.md#card-guard-stages-and-completion-criteria)
  now records it. The one gap it showed is the new `OPEN` row (below).
- `DEV#23` is `USER`, not `CARD`, because its next step is Matt typing
  Ctrl-C on the fake.
- `DEV#29` no longer says "after `DEV#28`", since the installer was built
  2026-11-01. It stays `DEFERRED` because release tags are a priority
  call, not ready work.
- `CARD#13`'s clause now matches its owner: only the model-offload seams
  remain (sequential was measured 2026-11-11), plus the other Z-Image crop
  checkpoints. `CARD#21` is trimmed under 500 bytes.
- `CARD#4`: [verified.md](../verified.md#quantization-and-offload-2026-10-25)'s
  Remaining paragraph now records the 2026-11-11 fp8/sequential load of
  Krea dev. The row became a `DECISION` (below).
- `GEN#46` keeps only the CI account's licences and the tokenizer move. Its
  shard maps join `GEN#65`, because the local account can read both
  repositories.

Real-world gates that the 2026-11-11 follow-up names are left without
rows on purpose, and are listed here so they do not drop silently: a
clean-machine install (`DEV#28`), HTTPS through a real proxy with a
trusted CA on the phone (`DEV#33`; to be folded into `UI#40` or `DEV#32`
stage 4 when Matt runs them), and idle release's real GPU and RAM
reclamation (`GEN#50`; beside `CARD#14` on the 3070).

## Method

One agent reconciled every row with its owner, the source and git history
since 2026-11-09. Five more proposed work from different angles:
composing, prompt help, grids and the queue; the gallery, reruns and
bundles; models, the loader and CPU slices of gated rows; defects in code
landed since 2026-11-03; and tooling, tests and docs. Each lens had a
separate verifier, told to refute every proposal: already built, decided
against, wrong evidence, a failure that does not trace, a Never-rated
feature, or work that needs the card. Of 26 proposals none was refuted.
Eight were reworded with corrected evidence, scope or status, and two
pairs were merged (the two `GEN#47` alternatives; the grid-values and Swap
tags autocomplete). [Reported use](../design/preferences.md) set the
ranking: Often features first, and real defects first among equals.

## Ready work (OPEN)

### PA#15 Check measures a grid's resolved prompts

Check measures the `{name}` template literally:
`prompt_encoding.measure(model_id, request.prompt, ...)` at
`backend/prompt_check.py:161-163`. `PromptAnalysisRequest`
(`backend/schemas_prompt.py:33`) has no grid fields, so a variable's value
can push one resolved prompt past the encoder's budget with no warning,
while the template itself looks short. B6 is rated Sometimes and C7 Often.
First step: add the encoder budget of each resolved prompt to the
`/api/grid` preview answer, since `web/grid-preview.js` already lists those
prompts. Widening the analysis request is the alternative. The server
decides, and the page only shows the counts and which values truncate.
Size M.

### PA#16 Tag help in Grid values and Swap tags fields

Tag autocomplete serves the prompt, the negative box
(`web/tag-lists.js:154`), the chip editor (`web/tags.js:503`) and Refine
(`web/refine-prompt.js:397`). It does not serve the Grid's Prompt variables
box (`els.gridPrompt`, `web/app.js:241-245`) or Swap tags' Find, Replace and
Add fields (`web/tag-swap.js` makes no `/api/tags` call). For example, on
Animagine, Swap tags `red hair -> silver_hiar` over 40 picked images queues
40 rerenders of a misspelt tag. B3 and C7 are Often. First step: attach
`scheduleEditorSuggestions` to both, then add an advisory unknown-tag
verdict per Replace or Add field (`/api/tags/validate`) that never blocks.
Size M for both together.

### GEN#63 A variable value naming another variable keeps its braces

`grid.substitute` (`backend/grid.py:358-377`) makes one pass over the
template, and `_check_variables` (`:414-421`) checks only the template. This
was reproduced on main: prompt `{a}, {b}` with `a = {b} hat | none` and
`b = red | blue` resolves to `{b} hat, red`, `{b} hat, blue`, `, red` and
`, blue`, so the literal braces reach the encoder. First step: decide
between refusing a value that holds a declared `{name}`, worded and coded,
and substituting it once more with cycle detection. Refusing is simpler
and keeps `CellKey` untouched. Add grid tests. Size S.

### GEN#65 Refresh model-downloads.json and check it with a tool

Seven recipes in `backend/model-downloads.json` have `"files": null`. Five
are noted "manifest unavailable without authentication" (FLUX.1-dev,
FLUX.1-schnell, FLUX.2-klein-9B and base 9B, Krea-2-Raw), and the two
FLUX 3 entries are noted "not readable" (2026-11-09). For each one
`model_info.download_reference_bytes` returns None (`:475-483`), so Models
shows its size as unknown (first-time F27). Several lite-infer sources
may also have moved since they were recorded, which makes their cache
state read "unknown". First step: write `tools/model_downloads.py`, which
rebuilds the file from the Hub at pinned revisions with Matt's local token,
and has a `--check` mode with no network. Then fill the seven recipes,
including FLUX 3's shard maps (moved here from `GEN#46`), and re-record any
moved source. Tests stay offline. Size M.

### GEN#66 Keep-awake under WSL says it holds when it holds nothing

`keep_awake.platform_inhibitor` (`backend/keep_awake.py:420-432`)
branches on `sys.platform` only. A server from `wsl/serve.sh` is `linux`
and takes `systemd-inhibit`, which blocks only the VM's logind, never the
Windows host's sleep timer. The status and the doctor can still report a
hold. Step (a), S, CPU only: detect WSL (`WSLInterop`, or `microsoft` in
the kernel release) and report it honestly, with a doctor WARN and a
sentence in usage.md. Step (b), S-M, after (a): a power request held from
Windows through interop (`powershell.exe`). The user check of (b) is
`powercfg /requests` on Matt's PC.

### DEV#55 A bad TLS pair fails start-up with "PEM lib"

`server_transport.setting_problem` (`backend/server_transport.py:37-43`)
checks only that each file exists. A non-PEM, mismatched or
password-protected pair reaches `uvicorn.Config` (`backend/main.py:1541-1559`)
and fails with an opaque `SSLError [SSL] PEM lib` traceback. Under the
systemd user service it then loops through restarts. This was reproduced on
CPU. First step: load the pair with `ssl.SSLContext.load_cert_chain` inside
`setting_problem`, with a password callback that raises, and refuse in
words through the existing start-up refusal path. A doctor row is optional.
Size S.

### GEN#67 A failed poll's retries count as activity

`keep_awake._POLLED_GETS` (`:79-83`) excludes `/api/status`, bundle status
and automatic backup. After a failure, `web/model-switch.js:934-937`
retries `GET /api/models` every 10 s, and that counts as activity, so an
idle tab after a refused catalogue read holds the PC awake. Since `GEN#51`
the same clock also delays idle release (`backend/main.py:485-490`). Only
unit tests guard the rule. The fake server runs with keep-awake off
(`tools/fake_server.py:135`). First step: exclude those retries, or make
them back off, then add a smoke whose fake records counted requests while an
idle page has every polling surface open. The smoke's row shares a full
port. Size S-M.

### DEV#20 The card guard keeps its sampling period on Windows

Stage 2 is recorded (see Reconciliation). The SDXL session showed that the
guard asked for 0.5 s sampling but sampled about every 1.8 s, because each
sample spawns `typeperf -sc 1` and `nvidia-smi` one after the other
(`tools/card_guard.py:358-365`), so its peaks can miss transients. First
step: use one long-lived `typeperf -si` stream (or PDH through ctypes) and
`nvidia-smi --loop-ms`, and report the achieved period in `summary`. Add
stub-host tests. CPU only. Size S-M.

### DEV#54 Job API tests race the fake's clock under -n auto

On 2026-11-12, `pytest -n auto -x` on main 798b2a0 failed
`tests/test_api_jobs.py::test_move_front_lists_the_authoritative_order_and_never_preempts`
once. It passed three times alone. The first job runs for about 0.35 s
(35 steps of 0.01 s, `tests/fakes.py:295-296`), and the test must make two
POSTs and a `/front` inside that window. First step: hold the first job on
an event instead of on a time budget. Audit `:105`, `:265`, `:287` and
`:325-337` in the same file as candidates, not known flakes. Size S.

### DEV#59 Seconds of margin for the tests' positive waits

AGENTS.md asks for seconds of margin on slow runners, and d516b59 raised
the rerun queue's guards from 5 s to 30 s after a 13 s Windows setup. These
are still 2-3 s: `tests/test_api_bundles.py:100` (`entered.wait(2)`),
`tests/test_api_expand_stop.py:65, :104, :138` (`result(timeout=2)`),
`tests/test_jobs.py:103, :145, :183` (`release.wait(3)`), and
`tests/test_download_progress.py`'s 100 x 0.02 s polls. First step: raise
positive waits (those expected to succeed) to 30 s, and keep negative waits
short. Then add a test that flags a new positive wait under 10 s. Size S.

### DEV#58 Conditions instead of fixed sleeps in the smokes

`tools/smoke*.json` holds 847 `{"wait": ms}` steps, 546 s in all, against
2,425 s for the whole hosted table in `tools/run_smoke_costs.json`: about 22%
of smoke time. smoke-analysis sleeps 61.8 s of its 86 s, smoke-lightbox
38.6 s of 66 s and smoke-animagine 33.3 s of 38 s. The ~470 s of sleeps
followed by a one-shot expect are race candidates to triage, not confirmed
races. Keep the sleeps that test a timer, a debounce or an absence. First
step: one PR for the five worst files, each sleep replaced by the condition
its next expect needs, with costs re-measured. Phase the rest by file.
Size M per phase.

### DEV#60 docs/rules.md sub-headings

`docs/rules.md` is 13,133 lines (930 KB). "### Frontend and CLI" runs from
line 3708 to 8473 and "## Tests" from 8473 to the end, with no subheadings
in either. AGENTS.md says to read an area's part in full before changing
it, and today that part is about 4,700 lines. First step: add `####`
sub-headings by component and smoke area, keeping the `## Layout` and
`## Rules` boundaries that `test_docs_maps` slices on. Then add a test that
fails when a section grows past a set length, skipping fenced code. Also
check whether the phone review's "Phones and tablets" anchor is a heading.
Size M.

### DEV#61 Fifth source-audit pass

Since the fourth pass's base (0351278, 2026-11-05) and 166 commits on,
`web/status.js` has grown from 871 to 1,067 lines, past its 1,000-line
trigger, and has no row in [source-audit.md](source-audit.md).
`backend/main.py` has grown to 1,569 lines (+201: idle release, passcode
status, storage and download routes), and `tests/test_linux_scripts.py` is
at 1,537. `web/lightbox.js` also grew. `web/app.js` (961) stays under its
trigger and is only monitored. First step: screen these four as the earlier
passes did. Split only where a seam exists, and record the files kept long.
Size M.

### DEV#62 A cost for every smoke in run_smoke_costs.json

`smoke-flux3`, `smoke-download-progress`, `smoke-phone-landscape` and
`smoke-phone-keyboard` have no entry, so `--shard` balances them at
`DEFAULT_COST = 12.0` (`tools/run_smoke.py:72`). `tests/test_run_smoke.py:672`
checks only that costs is a subset of the table. Some entries were hand-edited
after the 2026-11-03 measurement (for example 19b308b). First step: add the
four entries (hand estimates allowed until the next hosted refresh) and make
the test require an entry for every row. Size S.

## Matt's calls (DECISION)

Each has a recommended answer, which is not taken until Matt says so.

- **`GEN#64` grid sizes from presets.** `grid.parse_sizes`
  (`backend/grid.py:125-139`) accepts only typed `WxH`, and nothing writes
  into `#grid-size`. Do you sweep sizes? *Recommended:* a "+ panel size"
  button only, client-side with the server unchanged. Lowest priority of
  this set: C8 is Sometimes and the size axis is unrated.
- **`UI#51` the Steps/CFG/Mu disclosure.** UX#21's last candidate
  ([requirements](../design/requirements.md), [validation](../design/validation.md)
  item 2 of the next revision): A6-A8 are Sometimes and A9-A10 Never. Build
  it now as a reversible group like Custom dimensions, with effective
  values always in its summary, or leave it to the `DEV#1` trials?
  *Recommended:* build now. The trials can still judge it.
- **`GAL#25` prefix matching in gallery search.** `backend/storage.py:930-931`
  sends each free word to FTS5 as an exact quoted token, so typing `drag`
  finds nothing until `dragon` is complete, while the LIKE fallback
  (`:946-958`) matches substrings. The search runs as you type
  (`web/app.js:438`). *Recommended:* every free word as a prefix, with `-word`
  exclusions kept exact.
- **`GAL#26` delete a saved bundle.** `backend/api_bundles.py` has no delete
  route, and every Back up or Bundle & clear leaves a full archive in
  `bundles/`. *Recommended:* one file at a time behind a confirmation,
  refused while it is leased for download or import. Never automatic or
  bulk, which keeps `GAL#22`'s decline intact.
- **`GEN#47` lite-infer's renames.** Either (a) move the catalogue to the new
  names and keep old ids readable, or (b) keep writing the old ids and accept
  the new names as read-only aliases of the same entry and recipe (`launch._match`,
  `launch.py:503-512`, and `api_runs._model_name` match exactly today).
  Either way, add offline tests (an httpx `MockTransport`) of
  `hub_access.check` and downloads through a 307. Old ids in PNGs,
  `CellKey` and `run_id` never change. *Recommended:* (b), size S.
- **`UI#52` model favourites on the server.** `web/model-switch.js:71` keeps
  them in per-origin localStorage, so the LAN address, the HTTPS route and
  the phone each have their own set. *Recommended:* a revisioned per-gallery
  sidecar like saved searches, offering once to import this browser's set.
- **`GEN#6` the SDXL preview map only.** Re-asked: Matt kept `GEN#6` deferred
  on 2026-10-29. The new fact is that SDXL (Animagine) is now the first open
  checkpoint (`GEN#59`), so a token-less first visit sees no live preview
  (`model_info.py:539-546` maps SDXL to `AutoencoderKL/4`, which has no map).
  Fit that map on CPU, add the non-flow sigma rule with tests, and one card
  glance at Matt's OK. *Recommended:* yes, the SDXL half only. FLUX.2 stays
  deferred.
- **`INP#12` stage 1, desk half.** [plan-detailer.md](plan-detailer.md#stages-and-completion-criteria)
  bounds it to licence, provenance and wheel metadata for at most two face
  and two hand detectors, with no installs. *Recommended:* yes, as an `OPEN` S
  row. The CPU detection trial would stay a later decision.
- **`DEV#56` passcode renewal for an open tab.** Only `GET /` renews the
  30-day cookie (`backend/passcode.py:373, 385-402`), on purpose
  (`tests/test_api_passcode.py:351`). A tab or PWA left open and used daily is
  signed out at day 30. Should a counted API request from a session older
  than a day also renew it? *Recommended:* yes, size S, with the test's
  assertion changed deliberately.
- **`GEN#68` the model-cache inventory's next slice.** `GET /api/models/storage`
  (`GEN#55`, 2026-11-11) is raw JSON reached only by typing its URL, and its
  owner leaves the guarded delete "after this inventory has been considered"
  ([R9](review-2026-11-10-peers.md#r9-storage-view)). *Recommended:* look at
  your own output first, then a read-only "On disk" fold collapsed inside
  Models. Design the guarded delete only after that.
- **`DEV#63` AGENTS.md's Layout.** AGENTS.md is 54,844 bytes and loads into
  every session, while `docs/rules.md`'s Layout is the tested full map. Should
  its Layout shrink to one line per area pointing there, keeping the
  commands, "Rules that break things silently", the smoke/port table and
  the fake-knob line (`test_every_fake_route_and_flag_is_in_agents_and_rules`
  reads it)? *Recommended:* yes, with a diff to confirm that every cut line
  already exists in rules.md.
- **`CARD#4` (with `CARD#6`, `CARD#12`) under the Never ratings.** The
  2026-10-29 audit dropped "more quantization/offload card checks" because
  K4 and K5 are rated Never. These three rows remain. Keep them as card
  backlog, or retire them with their unmeasured modes recorded in
  verified.md? *Recommended:* retire them. Measure a mode only when it is
  the one Matt actually uses.
- **`GEN#15` was phase 3 met on 2026-11-11?** The FLUX.1 Krea dev
  sequential-offload session recorded peak pairs through the normal storage
  path, beside whole-card and shared samples
  (`docs/measurements/card13-flux-sequential-2026-11-11.json`). It was a
  sequential inpaint, not a model-offload text-to-image. Close `GEN#15` on it,
  or keep the row for a model-offload render? *Recommended:* keep it for the
  model-offload render, which is the mode the plan names.

## Matt's answers (2026-11-12)

Matt accepted every recommended answer above. As a result:

- Now `OPEN`, each with the scope its recommendation names: `GEN#47`
  (read-only aliases plus offline 307 tests), `GEN#68` (a read-only On disk
  fold; the guarded delete waits until Matt has seen it), `UI#52`
  (favourites as a server sidecar), `GAL#25` (prefix matching),
  `GAL#26` (a confirmed single-bundle delete), `UI#51` (the
  Steps/CFG/Mu fold), `DEV#56` (renewal on counted API requests),
  `GEN#6` (the SDXL map only), `INP#12` (stage 1's desk half only),
  `DEV#63` (the AGENTS.md Layout trim) and `GEN#64` (a "+ panel size"
  button).
- Retired: `CARD#4`, `CARD#6` and `CARD#12`. Their unmeasured modes stay
  listed in [verified.md](../verified.md#quantization-and-offload-2026-10-25).
- Kept: `GEN#15` stays `CARD` for a model-offload text-to-image render.
  The 2026-11-11 sequential inpaint does not close it.

## Added at Matt's request (2026-11-12)

Matt added two items after the answers above. Both were checked against
the code on main (e06ae8d); neither is built.

### GAL#27 Saved bundles listed by date

`bundles.list_bundles` (`backend/bundles.py:71-89`) sorts by file name,
descending. An app-named export happens to sort newest first. An uploaded
or renamed archive sorts by its name, though, so it lands wherever its
letters put it, whatever its date. Each entry already carries
`created_at` (the file's mtime). First step: sort by `created_at`, newest
first, with the name as the tie-break. Then check that the Bundles
dialog's list (and its phone sheet) shows the date the order follows.
Add a test with names whose order disagrees with their dates. Size S.

### UI#53 Models is blank while a bundle operation runs

`GalleryAdmission` (`backend/bundle_manager.py:192-207`) answers every
counted `/api/` path, `/api/models` included, with a 409 that has a
sentence and no `code` while a bundle import, export or clear runs. The
Models panel then opens empty, with no word on why. First step: give that
409 a refusal code (`refusal_codes.py`). Models then shows the reason in
place of the list, and reads again when the operation ends (through the
status poll that already reports bundle maintenance, not a new timed
poll). Check the other panels that read a counted route while one runs,
too. Add a smoke on the fake with a held import. Size S-M.

## Results (2026-11-12)

Each row was built on its own branch, then reviewed by a separate
adversarial reviewer, and every finding the fixer confirmed was fixed
before the branches were merged. Each branch passed the full suite, ruff,
the pinned type gate and its smokes. After the merge, the whole suite and
the smoke table passed again. Nothing ran on the card, and no weights were
downloaded. Plans with their own owners record their results there:
`GEN#48` in [plan-anima-refine.md](plan-anima-refine.md), `INP#12` in
[plan-detailer.md](plan-detailer.md), and `DEV#20` in
[plan-card-harness.md](plan-card-harness.md).

- **`GEN#63`** (done). `grid._check_variables` refuses a value that names
  a declared `{name}`, the value's own included, in words, in both
  variable modes. That covers the preview, Run grid and the CLI (exit 2).
  Braces that name no declared variable stay text. Refusing was chosen
  over a second substitution, so `CellKey` and run ids are untouched. The
  refusal carries no code: the page only shows it, as it does every other
  grid refusal, and `test_refusal_codes` admits only codes the page reads.
- **`PA#15`** (done). `/api/grid/preview` answers `budgets`
  (`backend/grid_budgets.py`): each distinct resolved prompt and negative,
  measured cold the way Check measures them, at most 128 of them, with one
  5 s tokenizer wait for the whole preview. A tokenizer that cannot be
  read stops the measuring at the first miss, so a gated repository costs
  one try per preview. The server names which encoders cut and which
  values tip a prompt over. The page puts the token range last in the
  listing, with a warning note for a cut. `EncoderReport` moved to
  `schemas_encoder.py`.
- **`GEN#64`** (done). A "+ panel size" text button on the Sizes label line
  appends the panel's `WxH` through `replaceBoxText`. A size already
  listed is refused in the hint. The label line keeps its height (asserted
  in smoke-grid), and on touch the button gets a 44 px press box.
- **`GEN#67`** (done). The page marks its own timed retries, and a
  debounced preview that only a poll caused, with `X-AuraDream-Poll`.
  `keep_awake.counts_as_activity` leaves those requests out. The fake
  gains `POST /api/fake/activity`, which records requests by the real
  rule. The new smoke-idle-activity asserts that an idle page holding
  every polling surface counts nothing, and it fails with the mark
  reverted.
- **`GEN#66`** (built; Matt's check left). Under WSL, `systemd-inhibit` is
  never used. With interop, `WslPowerRequest` holds a Windows
  `PowerRequestSystemRequired` from a `powershell.exe` child until its
  stdin closes. Without interop, the status, the start-up log and the
  doctor say nothing is held. A trial from real WSL 2 was accepted, and it
  released when its parent was killed. `powercfg /requests` needs an
  elevated prompt, so it is Matt's step.
- **`DEV#55`** (done). `server_transport.setting_problem` loads the TLS
  pair as uvicorn will (certificate first, then the key with a password
  callback that raises), so a bad pair is refused in words before
  anything binds. Settings refusals exit 78 (`SETTINGS_REFUSED_STATUS`),
  and the systemd unit gains `RestartPreventExitStatus=78`, which an
  installed service picks up when `install-service.sh` is run again. The
  doctor gains a TLS row.
- **`DEV#56`** (done). A valid session more than a day old is renewed on
  any request that `counts_as_activity` counts. Polls, streams, files,
  HEAD requests and Bearer callers never renew it, so an unused tab still
  ends with its month. A sign-out answered while a counted request is in
  flight holds that request's renewal back. The page-only assertion in
  `tests/test_api_passcode.py` was changed on purpose.
- **`DEV#54`** (done). `FakeBackend.hold` holds a call on an event. The
  move-to-front test and the file's other clock-racing tests (skips,
  pauses, cancels, the snapshot) hold their cells on it, not on a step
  budget.
- **`DEV#59`** (done). About 160 positive waits across 40 test files now
  allow 30 s. `tests/test_positive_waits.py` reads every test file's AST
  and fails on a literal under 10 s, unless the wait is under `assert not`
  or a timeout-expecting `pytest.raises`, or is a named negative.
- **`GEN#47`** (done). The app keeps writing the old lite-infer ids. The
  renamed repositories, which follow one rule (`_r32-bnb4-text-encoder`
  becomes `-r32`), are read as aliases through `launch.written_id`
  wherever an id enters. Offline `MockTransport` tests answer 307s under
  huggingface_hub's own client.
- **`GEN#65`** (done). `tools/model_downloads.py` has `--check` (offline)
  and a metadata-only refresh. All seven missing recipes are now filled:
  FLUX.1 dev and schnell about 33.7 GB each, FLUX.2 klein 9B and base 9B
  about 34.7 GB, Krea 2 Raw about 35.7 GB, and both FLUX 3 entries about
  20.3 GB, with their shard maps. The 18 lite sources were re-recorded,
  with `moved_to`.
- **`GAL#25`** (done). Every free word reaches FTS5 as a quoted prefix,
  while exclusions stay exact. The odd-quote LIKE fallback now matches the
  same way (`search.word_match`).
- **`GAL#26`** (done). Each saved bundle in Bundles has a Delete that asks
  first. `DELETE /api/gallery/bundles/files/{name}` refuses with
  `bundle_in_use` while the file is leased or an operation runs, and with
  `bundle_gone` when the file is missing. An archive the automatic backup
  journal counts is held, with its reason shown before any question.
- **`PA#16`** (done). Tag suggestions now work in Grid's variable values
  and in Swap tags' Find and Replace boxes. One `/api/tags/validate` call
  gives an advisory verdict on Replace and Add, and it never holds Queue.
- **`UI#52`** (done). Favourites live in `auradream.model-favourites-<db>.json`
  (`model_favourites.py`, `/api/models/favourites`), one set per gallery
  and canonical ids. Each press sends one star, in order. The picker
  offers once to import the browser's old set.
- **`GEN#68`** (built; Matt's look left). Models > On disk shows the
  storage inventory read-only. It reads when first opened and on Refresh,
  with no timed poll and no delete control. The fake plants a scratch Hub
  cache, and smoke-model-storage covers the fold.
- **`UI#51`** (done). Steps, CFG, Mu and True CFG are one fold,
  `#sampling-section`, whose one-line summary always shows the effective
  values. Only a press on the summary remembers the state. A refused
  number or Set CFG reveals the box. Shut, the fold puts Generate about
  62 px higher at 1400x900. smoke-sampling-fold is new.
- **`GEN#6`** (CPU half built; card glance left). `tools/fit_latent_preview.py`
  keeps the fitting recipe. On the same held-out images, the FLUX.1
  refit's R² of 0.966 reproduces the card fit's 0.964. SDXL's map, fitted
  from Animagine's VAE, has an R² of 0.909 (0.905 on Kohaku's and
  NoobAI's VAEs). `latent_preview.flow_view` turns non-flow latents into
  the flow form, so the threshold and blur stay one rule. The first frame
  is noisy (σ 8.85), so if the card glance finds it ugly, lowering the
  threshold for non-flow schedulers is a one-line follow-up.
- **`DEV#20`** (done). The guard reads `nvidia-smi --loop-ms` as one stream
  and the shared counters as one held PDH query. PDH was chosen over
  `typeperf -si`, which never sees an instance created after it starts. A
  local read-only run reached 0.5 s, and the summary reports
  `achieved_period_s`.
- **`DEV#62`** (done). The four missing costs were added, scaled from local
  times, and the test now requires the costs and the table to match
  exactly.
- **`DEV#58`** (phase 1). In smoke, smoke-analysis, smoke-animagine,
  smoke-lightbox and smoke-wide, fixed sleeps fell from 196.7 s to 34.4 s.
  Each one before an expect now polls that expect (`window.__when`), which
  `test_converted_smokes_poll_their_expects_instead_of_sleeping` enforces.
  The waits that remain are before actions, or absence checks that carry
  a `why`.
- **`DEV#58`** (done 2026-11-13). The other 54 smokes with a fixed wait
  before an expect were converted the same way, and their fixed sleeps fell
  from 383.1 s to 78.3 s. Most of what remains comes before taps and keys
  (smoke-phone-gallery's 500 ms gaps between picture taps, smoke-jobs'
  timed pauses), plus absences that carry a `why`. The test now covers
  every smoke file. Each smoke's file was converted by one agent and then
  checked by a second, adversarial one, which tried to show a new
  condition could pass on the state before its action. Every smoke passed
  at least three times in a row after its last edit, with the machine
  loaded by the others' runs. The reviews fixed conditions that did not
  name their action's end: smoke-passcode's status poll was read before
  its record, smoke-refine-sdxl's source could match an older record, and
  smoke-compare-cfg polled one axis of two. Reviews also tied smoke-tags'
  checks to the text just typed, waited out the 2 s card-note timer in
  smoke-gallery on its own clock, and anchored the scrim-tap clock in
  smoke-phone-gallery after the tap. smoke-offline's Try again now
  settles with the document it replaces. Found on the way and not fixed
  here: `model_favourites.read` (like the saved-search and snippet
  sidecars) opens its file with no lock, and on Windows a read that
  overlaps `os.replace` fails (a probe: 78 of 20,000 reads). That makes
  smoke-model-hover's last step flaky on Windows only. Hosted CI runs the
  smokes on Linux.
- **`INP#12`** (desk half done). Two detectors pass: YuNet (MIT) and the
  MediaPipe Hand Landmarker (Apache-2.0). Both are trained on photographs.
  The deepghs anime detectors stop at Ultralytics' AGPL-3.0 provenance.
  A CPU trial on illustrations needs Matt's call: accept photo-trained
  detectors, or commission a new bounded search.
- **`GAL#27`** (done). `bundles.list_bundles` sorts by the file's
  last-modified time, newest first, and breaks a tie by name. The dialog
  shows that date inside the size span, so no row gets taller on the
  desktop or at 393 px (a touch rule lets the size wrap beside Import),
  except a protected row on the phone: beside `GAL#26`'s Delete it takes
  one more line (+51 px), which smoke-phone-dialogs allows. smoke-bundles
  and smoke-phone-dialogs measure each row with and without its date.
- **`UI#53`** (done). `GalleryAdmission`'s 409 is coded
  `gallery_maintenance`. While an import, export, backup or clear runs,
  Models shows the reason in place of the table, holds Switch, and keeps
  the table's height so On disk stays put. On disk, Check and the tag
  finder say the same. A refused catalogue or On disk read is repeated
  through the existing bundles status poll once the operation ends, marked
  as a poll. The fake gains `/api/fake/bundle` (`hold`), and the new
  smoke-gallery-maintenance covers the hold. Places still showing the
  server's sentence (Finish run's hold title, Expand, the critic, the tag
  helper's suggestions, previews) are listed in the design notes.
- **`DEV#61`** (done). The fifth pass of
  [source-audit.md](source-audit.md) made five splits: the confirm
  questions into `web/ask-dialog.js`, the server-held model stars into
  `web/model-favourites.js`, Expand's state and the translation writers
  into `web/prompt-expand.js` (the page loads 78 scripts), and
  `tests/test_linux_scripts.py` and `tests/test_keep_awake.py` into helpers
  plus separate files, with the same test ids. Nineteen long files are kept
  whole, each with a dated reason and a trigger; `main.py`'s fired
  `/api/models` trigger was re-screened and its seam kept.
- **`DEV#60`** (done). `docs/rules.md`'s Frontend and CLI, Tests and Layout
  sections have headings by area (Phones and tablets one of them, with its
  own parts), moved, never reworded. `tests/test_rules_sections.py` caps a
  section at 350 non-blank lines outside fenced code; four Rules sections
  are held at their current lengths, and their numbers fall with them.
- **`DEV#63`** (done). AGENTS.md's Layout is nine lines, each linking to
  its part of rules.md's Layout (55,797 to 36,593 bytes). Five facts found
  only in AGENTS.md moved to rules.md first. `tests/test_docs_maps.py`
  keeps the short Layout linked to every part and under 2,500 characters.
- **`GEN#48`** (stage 1 done). Private Img2Img passes for Turbo, Aesthetic
  and Preview are in `backend/backends/anima_refine.py`, with CPU tests.
  Nothing is wired in, and every Anima profile still refuses Refine. Next
  is the plan's stage 2, a card qualification.

## Not added

None of the 26 was refuted. Some were folded in rather than given rows of
their own: the second `GEN#47` alternative and the reconciliation's 307 test
slice (both in `GEN#47`), `GEN#46`'s shard maps (in `GEN#65`), and the
grid-values autocomplete (in `PA#16`). The activity smoke is part of
`GEN#67` rather than a tooling row.

## Validation

See the commit. Links in the changed files were checked, and the docs-map
tests and ruff were run. No code changed.
