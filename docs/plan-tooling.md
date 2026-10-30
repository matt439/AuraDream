# Tooling work ready for implementation

Owner for `DEV#4`-`DEV#7`. The first three came from the
[2026-10-25 review](review-2026-10-25.md#tooling); `DEV#7` was found while
auditing the queue on 2026-10-26. The audit established this plan; dated
implementation and qualification results below record the later work.
The dated sections at the end own later tooling rows: `DEV#9` (2026-10-28)
and `DEV#24`-`DEV#26` (2026-10-29 TODO audit).
Work in the order below. All stages use CPU, synthetic images and scratch
galleries; they need no model weights or real gallery contents.

## Portable smoke screenshots

`DEV#7` · implemented and qualified on Windows, 2026-10-26.

**Evidence.** `tools/smoke-compact-setup.json` has eight `shot` destinations
under `C:\Users\matsi\AppData\Local\Temp` (lines 44, 71, 97, 124, 150,
177, 203 and 230 at this audit). `tools/browser.mjs` writes `shot`,
`fullshot` and the destination in `elshot` literally with `writeFileSync`.
An account without that Windows directory cannot save those screenshots;
on Linux, the Windows path becomes a filename in the working directory.
This finding concerns output paths; Linux assertion behavior is unverified.

**Bounded change.** Give repository smokes portable relative screenshot
names and one documented output-root policy. The browser harness must apply
the same policy to `shot`, `fullshot` and `elshot`. The smoke runner gives
each smoke an isolated output location, reports it, and permits CI to collect
the artifacts from a caller-selected root. Preserve failure artifacts and
keep parallel runs from overwriting each other's screenshots.

**Done and checked.** No screenshot destination in a repository smoke is a
Windows or POSIX absolute path or escapes its output root. Inspect only the
three screenshot fields: fixture messages such as `C:\\gallery\\outputs`
in `smoke-bundles.json` are legitimate test data. Focused checks exercise all
three screenshot forms and output-root creation, with a deliberate bad
destination rejected by the repository-smoke check. Run `smoke-compact-setup`
on a scratch fake and verify its eight files appear under the reported root.
Run affected runner/profile tests and docs-map checks. Record the platform
actually checked; Linux behavior is qualified in `DEV#4`.

**Result (2026-10-26).** All eight compact-setup destinations now use
relative names. `browser-output.mjs` validates every screenshot field and
creates parent directories for `shot`, `fullshot` and `elshot`; absolute,
drive-relative, traversal, Windows device and otherwise nonportable names
are refused before launch. `AURADREAM_SCREENSHOT_DIR` chooses the
root; a direct call without it reports a fresh system-temp root.
`run_smoke.py --artifacts PATH` creates separate run/smoke directories,
reports them and retains screenshots even after failure or successful
gallery cleanup.

Focused output/runner/docs-map tests: **50 passed**. Real browser profile
and screenshot-form tests: **4 passed, 1 Windows SIGTERM skip**, including
three PNGs retained after a deliberately failed assertion. The compact
smoke passed against its isolated fake (**24 seconds**), writing all eight
390/600/1400/2048 closed/expanded images below its reported root. Node
24.14.0 and Edge 156.0.4314.42 were used on Windows; no weights or GPU work.
Browser starts failed inside the execution sandbox before any page step;
the supported, approved outside-sandbox run passed. Linux remains the
separate `DEV#4` qualification. Focused ruff and both Node syntax checks
passed.

## Shared classic-script scope

**Built and checked 2026-10-26 (`DEV#5`).** Independent of `DEV#7`.

**Evidence.** `web/index.html` currently lists 52 local classic scripts;
`app.js` is last and owns the wiring and boot. `tests/test_docs_maps.py`
checks the script count and module map, not JavaScript syntax or duplicate
bindings. The [classic-scope design contract](design-notes.md#the-page-scripts-share-one-classic-scope)
requires top-level bindings to remain shared. `app.js` itself declares
bindings, including `searchTimer`, `galleryHead` and `TOKENS_AIMED`, so
excluding it entirely would miss collisions.

**Implementation.** `tools/check_classic_scripts.py` uses the standard
library's HTML parser to derive every script from `index.html` in document
order. Missing files, inline/module/remote scripts, changed loading attributes
and paths outside the web directory fail explicitly. There is no second
script inventory. Run it directly with:

```powershell
uv run python tools/check_classic_scripts.py
```

The Node helper parses each original with `vm.Script`, then instantiates its
declarations in one shared empty context. A leading throw stops before any
source statement or initializer, including `app.js`'s boot. It preserves each
script's own strictness: a parser-only appended `with` probe distinguishes
strict directives from comments or ordinary strings. This keeps Annex B's
sloppy block-function declarations distinct from strict block functions,
and lets V8 decide lexical/var/function collisions rather than approximating
declarations with a regex or concatenating scripts. Errors name the failing
file and the preceding files in order. This is a load-time guard, not a DOM
emulator or a replacement for browser smokes.

**Checks.** On Windows with Python 3.13.15 and Node 24.14.0, all 52 real
scripts passed. `tests/test_classic_scripts.py` and `tests/test_docs_maps.py`
passed all 35 cases, including original syntax errors, lexical collisions
with `app.js`, permitted var/function sharing, strict/sloppy block functions,
missing/changed script kinds and a missing Node executable. Ruff passed.
The initial pytest run hit an inaccessible existing shared temp directory;
the passing run used a fresh task-owned `--basetemp`, with the cache plugin
disabled. The ordinary `smoke-flux` browser check also passed on an isolated
fake gallery (17 s browser, 24 s total; no real model), using the supported
sandbox approval route for browser startup. Linux and hosted CI execution
remain unmeasured.

The CPU CI job verifies Node before installation/tests; pytest runs this same
guard and fails if Node is absent. There is no local skip policy: install
Node to run the complete CPU suite. No npm dependencies or build step were
added.

## Browser smokes in CI

`DEV#4` · done 2026-10-28. Stages 1-3 are built and qualified: the hosted
`browser` job passed the whole table on main after PR #15 merged (below).

**Initial evidence (2026-10-26, before the stages below).**
`.github/workflows/ci.yml` had an always-run `lint` job and a conditional
CPU `test` job; documentation-only changes skipped the latter. There was
no browser job. `tools/run_smoke.py` already read the authoritative `AGENTS.md`
table, isolated each fake's gallery, offset parallel workers' ports and
retained a failed fake's log in its temporary directory. It lacked a stable
CI artifact report. `tools/browser.mjs` knew Linux Chrome paths and used
Node's native `fetch` and `WebSocket`.

**Stages.**

1. Prepare diagnostics: a predictable report/artifact root containing each
   smoke's result, duration, browser output and failed fake's log. Include
   fake-start failure and browser failure; preserve the failed exit status
   and terminate owned fake/browser processes. Use the existing runner and
   port table, not a second CI smoke inventory.
2. After `DEV#7`, run representative browser/layout smokes on Linux Chrome,
   then the complete table. Record the actual browser/Node versions, fonts,
   concurrency, elapsed time and resource behavior. Check native
   `fetch`/`WebSocket` availability explicitly. Fix font-sensitive assertions
   by preserving the behavior they prove; do not loosen thresholds merely
   to turn a failure green. Keep existing Windows coverage valid.
3. Add a separate browser job with read-only permissions and an explicit
   timeout/concurrency justified by those runs. Upload failure diagnostics
   even when the job fails. It must not hold up or replace the CPU job.

**Done and checked.** Focused runner tests prove failure reporting,
artifact retention and cleanup. The full Linux Chrome table passes inside
the selected budget, and a deliberate failing smoke produces downloadable
diagnostics and a failed job. Record actual qualification evidence here
and in the tests documentation. A workflow file alone does not close this
row; hosted execution remains unverified until it has run.

**Stage 1 result (2026-10-26).** `run_smoke.py --artifacts PATH` retains
`summary.json` in its unique run directory and `result.json` plus
`browser.log` in every smoke directory. Each result records the smoke,
checkpoint, effective port, phase, pass/fail result, exact exit code and
phase duration. Early port, fake-launch/readiness and browser-launch
failures get receipts too; startup exception text is in both the detail
and log. Each failed smoke receives `fake-server.log` after the owned fake
stops. Successes retain their result/output/screenshots after the scratch
gallery is removed. Existing table-derived setup and parallel isolation
remain the only smoke inventory. Both serial and parallel browser output
are now captured together and printed after the smoke finishes.

Interruption gives the Node harness time to run its existing browser
cleanup, with a Windows owned-tree fallback, and stops/waits for the fake
(including terminate-to-kill fallback). An interrupted smoke retains its
complete partial output and an exit-130 receipt; interruption prevents
the final run summary, so collect the per-smoke receipts in that case.

Focused runner/output/docs-map checks: **63 passed**; focused ruff passed.
Tests cover original exit codes including script refusal, launch/readiness
failures, log retention, isolation, successful gallery cleanup and both
cleanup fallbacks. The interruption regression retains all 60 log lines.
A deliberately failing synthetic fixture injected through the existing
FLUX row on an isolated fake returned runner exit 1 and retained a failed
result/summary, full browser output, nonempty fake log and valid screenshot;
the fake port was closed after completion (**7.4 seconds**, Windows, Node
24.14.0, Edge 156.0.4314.42). This was an intentional fixture failure, not
a failure of the repository's FLUX smoke. The fixture and mock checks use
no model weights or GPU generation.

**Remaining qualification gate.** Linux Chrome's representative/full
table, resource budget and hosted diagnostic upload are unverified.
`wsl.exe --list --verbose` and `wsl.exe --status` were denied with
`Wsl/EnumerateDistros/Service/E_ACCESSDENIED`; the denied access was not
retried through another route. No Linux browser CI job has been added.

**Native API preflight (2026-10-26).** The harness now checks that Node's
`fetch` and `WebSocket` globals are callable before browser discovery,
script reads, profile creation or screenshot/output work. Missing or
noncallable APIs return a fixed diagnostic naming only the missing APIs
and exit 2; a Node version number is not treated as capability evidence.
Fourteen black-box tests mask either or both APIs and instrument the real
filesystem/process/network entry points to refuse downstream effects.
They also verify that callable globals continue into ordinary browser
discovery. Node syntax and focused ruff pass. The unchanged supported path
passed all 35 checks in `smoke-automatic-backup` against an owned Windows
fake (Node 24.14.0, Edge, 14 seconds total). No startup export, credential
lookup, model load or GPU work occurred. This closes the source preflight
gap in stage 2; Linux and hosted qualification remain open.

**Hosted CPU fixture correction (2026-10-27).** Node 22.23.3's ESM loader
reads module source through `fs.readFileSync`, so the original capability
fixture trapped the loader before the harness could check its APIs. The
fixture now captures the two real module bodies before instrumentation and
loads them as data URLs, changing only their local dependency specifier.
There are no filesystem exemptions: six negative controls still refuse
input/module reads, output writes, process creation and network APIs. The
cold capability/diagnostics/output/docs-map group passes all **66 checks**
on Windows with Node 24.14.0. This corrects fixture portability; it does not
qualify browser startup, layout or profile cleanup.

**Owned-browser diagnostics (2026-10-26).** Set
`AURADREAM_BROWSER_DIAGNOSTICS=1` for a scratch smoke to retain at most
16,384 decoded stderr characters and report its owned exit/signal on failure.
Normal runs stay quiet; browser/security flags, browser choice and profile
protocol are unchanged. A synthetic child-stream check exercises bounding and
failure framing without launching a browser or network request. The focused
cold diagnostics/capability/output/map group passed **60 checks**.

The performance-selector smoke first lost its DevTools connection before any
step in two normal attempts. A diagnostic run then reached 74 steps and produced
five desktop/narrow screenshots, all inspected; one fixture assertion used the
wrong reason phrase. After that fixture was corrected, the fresh browser exited
before step 1 with `IAppCapability` access denied (`0x80070005`), repeated child
exit `-1073741790` and `GPU process isn't usable. Goodbye.` Its own-profile cache
also reported sharing error `0x20`. The exact failed command was the ordinary
`run_smoke.py smoke-execution-selector --jobs 1` with diagnostics and fresh
scratch/artifacts, not model generation. Receipt:
`work/execution-selector-ui-a3.txt`.

All browser launches stopped after that actual Windows denial. No flag/browser
substitution, Windows security change or alternative browser API was tried.
Corrected end-to-end selector/profile/regression coverage remains unverified;
the source and five inspected earlier screenshots do not establish a full pass.
Linux/hosted qualification still has its separate recorded WSL/publishing gates.

The first broad pytest request excluded only the real-browser profile file and
was rejected before execution by automatic approval review over possible browser
launches in the remaining group. The approved safer CPU gate excluded every
`tests/test_browser*.py` file plus `tests/test_run_smoke.py` and passed **5,490
tests, seven skips**. The separate **60** cold Node/output/map checks above had
already passed with no real browser. Neither result closes actual browser/profile
acceptance or authorizes retrying the Windows denial.

**Windows browser access restored (2026-10-27).** An ordinary
`run_smoke.py smoke-flux` launched Edge 156 and passed, with no flag, browser
or security change. The whole table then ran (`--all --jobs 4`): 59 of 76
passed. Sixteen failed again when run alone. Triage found four app regressions
made while the browser was blocked: the Step cache row above Generate, Expand's
size refusal focusing a field inside a closed fold, "Expand…" wrapping Reset,
and Anima Aesthetic withholding a valid reorder. The other twelve were smokes
not updated for deliberate changes: click to open, modal-owned keys, the queue
front and Skip buttons, Size as a sheet axis, prose chips and the closed
conditioning fold. After the fixes the whole table passed (74 of 76 in one
`--jobs 3` run; the other two passed alone, one of them a load flake), as did
`pytest -n auto` with the browser tests included (6,170 passed, eight platform
skips). That run included `smoke-execution-selector`, `-reuse`,
`smoke-automatic-backup`, `smoke-retention` and `smoke-credential`, the Windows
browser checks the denial had left unrun. Linux Chrome and hosted CI (`DEV#4`)
remain unqualified: the WSL denial was not retried.

**Linux Chrome and the browser job (2026-10-27, stages 2-3).** Run in
a cloud container (no GPU, no weights) as a non-root user, since Chrome
refuses to start as root without `--no-sandbox` and the harness's flags were
not changed: Chrome for Testing 153.0.8010.12, Node 22.22.0, 4 cores,
`run_smoke.py --all --jobs 3`, about 12 minutes for the 77 rows. Native
`fetch`/`WebSocket` passed the harness's preflight.

- With the container's own fonts 59 of 77 passed. Almost every failure was
  text measured to fit: the page's stack is `"Inter", "Segoe UI", system-ui`,
  Windows draws it in Segoe UI, and Linux drew it in Inter (installed there)
  or DejaVu Sans, both wider.
- `tools/smoke_fonts.py` installs Microsoft's Selawik (Segoe UI's open
  replacement, pinned by SHA-256) as Segoe UI through a fontconfig file and
  hides Inter. A family alias alone did not work: Chrome rejects a fontconfig
  substitute whose name differs, so the scan-time rule renames the family.
  Selawik's hhea line metrics are 1.2 em where Windows uses its usWin 1.33 em
  pair, which shortened every `line-height: normal` box (the adapter select
  in `smoke-wide`); the installed copies carry the usWin pair in hhea.
  With these, 71 of 77 passed locally.
- Two smokes assumed the Windows host or an idle machine, and were fixed:
  `smoke-model-switch` expected the fake's Windows timing basis (the fake
  answers for `sys.platform`; on Linux a Windows FP8 timing is "timed on
  Windows"), and `smoke-more-seeds` read its count line and Cancel's title
  without waiting, where under load the fake's timings give an "about 0s"
  estimate and the first job can end before the title is read.
- The `browser` job (`ci.yml`) runs the table on the hosted runner's
  `google-chrome`, sandboxed as the runner user, after `smoke_fonts.py`, and
  uploads `smoke-artifacts` on failure. Its first hosted run passed 73 of 77
  in 778 s and uploaded the artifact (PR #15).
- Still failing on both: `smoke-lightbox` (the 600 px letterbox room is 77 px
  where it expects over 100: the head wraps onto a row of its own below the
  actions), `smoke-select` (the QHD count's tail is 0.85 px short) and
  `smoke-control-sizing` (ERNIE's Reset wraps under the Prompt caption by
  about a pixel). Selawik matches Segoe UI closely, not exactly, and symbols
  Segoe UI lacks fall back to DejaVu Sans on Linux, much wider than Segoe UI
  Symbol on Windows (`⌦` measures 1.41 em). `tools/font-widths.json` logs
  every character the page draws, in the font Segoe UI resolves to.
- Matt ran it on Windows (Edge 156, 2026-10-27), frozen as
  `tools/segoe-widths.json`. Against it Selawik's Regular and Bold matched
  every ASCII advance; its Semibold was off by up to 0.16 em, and the symbol
  fallbacks widely (`↑` +0.37 em, `⏸` -0.5 em). `smoke_fonts.py` now sets each
  copied font's advances to Windows' widths (Selawik at 400, 600 and 700,
  DejaVu Sans and Bold, Noto Color Emoji). Re-measured on Linux, every
  character matches but `🎲` (Noto Color Emoji takes its advance from its
  bitmaps, not hmtx), `⌕`, `⤢`, `⏲` and `⏸` (no copy has them as text, so
  another system font draws them), and `−`/`≈` at 600, which DejaVu Bold
  serves with Windows' 700 widths, 0.013 em wider. None of these is in a
  fit a smoke measures.
- Hiding every font but the copies passed locally (Chrome for Testing, 77
  of 77) but on the hosted `google-chrome` the browser process aborted on a
  character no font had (`SkFontMgr_FontConfigInterface ... Not
  implemented`), and each smoke it hit waited out its 300 s, past the job's
  45 minutes. Only Inter and the system's own copies of the two fallbacks
  are hidden now.

- With that, the hosted `browser` job passed the whole table, 77 of 77, in
  about 13 minutes on both the push and pull-request runs (PR #15). Earlier
  hosted runs' real failures already showed a failing smoke failing the job
  with downloadable diagnostics. The merge commit's own run on main
  (ab832cf, 2026-10-27 23:24 UTC) passed `browser` in 13 minutes beside
  `lint` and both `test` jobs, which closed `DEV#4`.

## Backend type checking

`DEV#6` · implemented and qualified locally, 2026-10-26. The pinned gate
is configured in CPU CI; hosted execution remains unverified.

**Evidence.** `backend/` uses annotations across schemas, applied requests
and storage records. The 2026-10-26 audit did not install or run Pyright.
Stage 1 was implemented later that day: `pyrightconfig.json` selects basic
mode, Python 3.13, all platform branches and the local `.venv`, over every
Python source in `backend/` and `tools/`, with no diagnostic overrides or
module exclusions. Pinned Pyright 1.1.407 checked 104 files in 10.4 seconds:
334 errors in 52 files, zero warnings. The
[baseline report](typecheck-baseline-2026-10-26.md) records the command,
environment, complete per-rule/per-module counts, third-party interface
gaps and eight bounded investigation batches. No app import, model load
or GPU run was used for static checking. Later stages fixed the complete
scope and added the locked development tool and CI command below.

**Batch 1 (2026-10-26).** Benchmark keyword/format contracts are checked
without changing retained timing values or selection behavior. All nine
benchmark diagnostics are gone; the full scope now has 325 errors in 104
files, zero warnings. The 21 existing benchmark tests and focused ruff
pass, and independent review found no blocking issue. No suppressions or
module exclusions were added. The baseline report preserves the initial
counts and records later batches separately.

**Batches 2 and 7 (2026-10-26).** CLI/launcher/lifecycle and
prompt/catalogue contracts resolve another 15 and 22 initial diagnostics.
Their scoped static checks are clean, with no suppressions or casts.
Existing focused checks passed: 274 CLI/storage/API/lifecycle tests and
488 prompt/tag/critic/catalogue tests. The final stdout correction passed
its 13-test subset and ruff. Both batches received independent review.
The baseline report records the exact scope and receipts; this does not
claim the entire configured type-check scope is clean.

**Batch 3, schema/PNG slice (2026-10-26).** The scoped checks address 17
initial findings. All 31 generated model schemas and existing custom
errors are unchanged; 245 schema/refine/snippet tests and 17 PNG API tests
pass. Independent reviews accepted the changes. Two precise, locally
explained Pydantic typing exceptions preserve the supported computed error
code and required-field override; the baseline report records both.

**Batch 4, keyword slice (2026-10-26).** Precise Hub/recipe keyword
contracts remove 40 findings from four converter modules, leaving 24
library-export/model-interface findings there. Pins, checksums, filters
and cache serialization are unchanged. The 111 mocked loader tests,
focused ruff and independent review pass; no suppressions or casts.

**Batches 6 and 8 (2026-10-26).** Storage/bundle/job and developer-tool
boundaries resolve another 18 and 25 initial findings. Their scoped checks
are clean without suppressions or casts; 237 storage/bundle/job tests and
63 tooling/docs-map tests pass. Bounded synthetic fixtures cover the tools
without real scale/interruption benchmarks, downloads or model work.
Independent reviews found no actionable issue.

**Batch 3, route/grid slice (2026-10-26).** Six modules resolve another
57 initial findings with explicit existing invariants and unchanged
response validation. The 480 focused grid/size and HTTP/rerun checks,
scoped static checks and ruff pass. Independent helper and route reviews
found no actionable issue; no suppressions or casts were added.

**Remaining loader interfaces (2026-10-26).** The converters' remaining
24 findings and loader helper's 21 findings are addressed. Scoped static
checks, ruff, 111 converter tests and 47 loader/runtime tests pass; three
Linux-only map checks skip on Windows. Independent review accepted the
same-class imports and explicit return/keyword contracts. Four local
comments document verified lazy-export/bound-method typing gaps; the
quantization and RAM-release behavior is unchanged.

**Batch 5 (2026-10-26).** All 63 initial pipeline/schedule/refine/preview
findings are addressed. The 387 focused backend tests and 53 tiny custom
pipeline tests pass; 24 mask mode/block cases match prior bytes. Scoped
checks, ruff and independent reviews pass. Seven local custom-pipeline
sites document verified library annotation gaps; the main backend needs
no exception. Schedules, seeds, arithmetic, source composition and cleanup
remain unchanged; no GPU behavior was measured.

**Critic lifecycle slice (2026-10-26).** All 23 runtime/server findings
are resolved without suppressions or casts. Scoped checking, ruff and
independent review pass; 62 focused tests pass with one expected skip.
A sandboxed invalid-executable test hung, its owned session was closed,
and the isolated test plus full focused suite passed through supported
approval. No process-inventory denial was bypassed, and no real critic
or GPU work was used.

**Stages.**

1. Select and record a reproducible Pyright version and basic-mode
   configuration for Python 3.13 over `backend/` and `tools/`. Run a baseline
   without loading weights or importing modules for their runtime side
   effects. Record counts by diagnostic and module, required third-party
   typing gaps, and bounded fix batches. This baseline and batch plan are
   the first deliverable; do not promise a size for the unknown fixes.
2. Fix one coherent batch at a time, preserving runtime behavior. Address
   real defects with focused regression coverage. Each unavoidable
   suppression names its diagnostic and explains the limitation locally;
   no blanket ignores, whole-module exclusions or unexamined `Any` casts
   to obtain a clean result.
3. Add the reproducible checker to the development instructions and CI
   beside ruff once the complete configured scope passes. Missing tools or
   an empty scope fail the gate.

**Done and checked.** The complete configured scope passes basic checking,
remaining individual suppressions have reasons, and CI runs that same
command. Run ruff and tests appropriate to any behavior changed while
fixing diagnostics; finish with the required CPU suite. Update this plan
with the baseline, completed batches and actual checks. No GPU acceptance
claim follows from a clean type check.

**Final result (2026-10-26).** The actual Python-wrapper gate checks
**105 files with zero errors and zero warnings** in 9.476 seconds. Its
first whole-scope run caught ten cross-module contract findings beyond
the scoped batches; explicit PNG dimension invariants and the shared
pinned-repository input type resolved them. The locked development group
adds only Pyright 1.1.407 and nodeenv; existing package versions stay put.

`tools/typecheck.py` verifies the local package/bundled JavaScript version
and exact platform Node executable before invoking the wrapper. Its
documented controls prevent update queries or automatic npm/Node
provisioning. Missing tools, version mismatch, empty configured/actual
scope and malformed output fail; diagnostic output/status survive.
Independent review caught and verified the Windows command-shim case;
29 focused gate tests cover it and the other boundaries. Gate/docs/CI
focused checks passed all 87 tests, and repository ruff passed.

The final offline CPU suite passed **3,950 tests, eight skipped**, in
**186.90 seconds**, using two workers with file-based distribution on
Windows/Python 3.13.15. The skips were one unavailable symlink check,
three Linux critic-process cases, three `/proc/self/maps` checks and the
Windows-unsupported catchable-SIGTERM browser case. No model download,
GPU render or human acceptance was performed. The CI job runs the same
gate after install; docs-only changes retain their lint-only path. Linux
and hosted execution were not run or claimed; no push/PR was made.


## Explicit fixture prerequisites (2026-10-28, DEV#9)

The [dated review](review-2026-10-28-fatal-prerequisites.md) owns the optional
fatal expectation, cold metadata checks and actual ordinary-driver cleanup
qualification. This is a bounded step-control improvement; it changes no
browser discovery, launch flags, profile or security path and retries no older
denied operation. Ordinary eval/nonfatal aggregate behavior stays compatible.

## Smoke reruns alone (2026-10-29, DEV#24)

**Status:** `DEV#24` done 2026-10-30: both stages built in one commit.
`tools/run_smoke.py` takes `--rerun-failed-alone` and `--from-summary PATH`
as designed below; `tests/test_run_smoke.py` gains the stage 1 cases (only
exit 1 reruns, serially at the base offset; `passed-alone` with exit 1;
each bad `--from-summary` input refused with exit 2 before `run_row`;
nothing to rerun exits 0), and the existing runner tests pass unchanged.
A real `--from-summary` over a hand-written summary naming `smoke-flux`
reran it alone on a fresh fake (`passed-alone`, exit 1).

Taken while Matt was away (2026-10-29), revisit if wanted: the design left
open what a `--from-summary` run writes and exits. Its summary has
`results: []` (this run had no first pass), `jobs: 1`, its `offset`,
`from_summary` (the path as given), `rerun` and `verdicts`; it exits 1
whenever anything was rerun, since the source run's failures stand, and 0
only when nothing was. Rerun receipts' `artifact_dir` is relative to the
run root (`rerun/<dir>`), so the `results` glob (`*/result.json`) never
sees them. A rerun that ends in any exit other than 0 (including a port
refusal) is `failed-alone`; its receipt keeps the exact code. Duplicate
names in a source summary rerun once.

Review fix (2026-10-30): the first pass's `summary.json` is now written
before any rerun and rewritten after it, so a Ctrl-C during the reruns no
longer loses the completed first pass's receipts; `duration_seconds` and the
console's "passed in" time stay the first pass's, not the reruns' too.

### Evidence

`docs/rules.md` (Tests, the sentence after "`tests/test_run_smoke.py`
holds every smoke to one parsed row") says a timing-sensitive smoke can run slower under parallel rows, and "a failure
seen only under `--jobs` is rerun alone before it is believed". Today that
rerun is by hand: read the failed names from the console's `failed:` line
or `summary.json`, then run `tools/run_smoke.py <names>` again.
`tools/run_smoke.py` already runs only the named smokes, each row on a
fresh fake with its own new `--data` (`run_row`, `tools/run_smoke.py:184`;
the fake's arguments at `:65`), so the missing part is only
choosing the names and recording both answers side by side.

`summary.json` (`tools/run_smoke.py:356-358`) holds `duration_seconds` and
`results`, one receipt per smoke from its `result.json` (`smoke`, `model`,
`port`, `phase`, `result`, `exit_code`, `duration_seconds`, `artifact_dir`).
It does not record the `--jobs` or `--offset` the run used, so a later
reader cannot tell a parallel failure from a serial one.

### Design

- `--rerun-failed-alone`: after the planned run, each smoke whose exit was
  1 runs again alone, serially (one worker, the run's `--offset`), on a new
  fake and a new artifact directory below the same run root. Exit 2
  (a refused smoke file or command) is never rerun; neither is an
  interrupted run, which writes no summary.
- `--from-summary PATH`: read an earlier `summary.json`, take the receipts
  whose `exit_code` is 1, and run those smokes alone as above, in a new run
  root. It refuses with exit 2, before any process starts, when PATH is
  missing or malformed, names a smoke no longer in the port table, or is
  combined with smoke names or `--all`. With nothing to rerun it says so and
  exits 0.
- `summary.json` gains top-level keys only: `jobs`, `offset`, and, when a
  rerun happened, `rerun` (its receipts, in the same shape) and `verdicts`,
  one per first-pass failure: `failed-alone` or `passed-alone`.
  `--from-summary` also records the source path. `results` keeps its
  current shape exactly; `test_main_writes_a_collectable_summary_and_keeps_failure_status`
  (`tests/test_run_smoke.py:323`) still passes unchanged.
- The exit code is the first pass's. A smoke that passes alone is reported,
  not believed: the run still exits 1. The console prints the verdicts after
  the existing `failed:` line.
- Each rerun keeps the existing guarantees: fresh `--data`, its own
  screenshots, `result.json`, `browser.log` and, on failure,
  `fake-server.log`; port refusal before any start
  (`_port_plan_error`); cleanup on interruption.

### Non-goals

- No flag that turns a `passed-alone` result into a pass, in CI or locally.
- No retry of the same smoke under `--jobs`, and no automatic retry count.
- No change to the port table, the smoke files or `browser.mjs`.
- The CI `Smokes` step keeps its command; whether CI uses
  `--rerun-failed-alone` is decided after `DEV#25` gives Linux timings
  against the job's 45-minute timeout.

### Stages and completion criteria

1. **Runner.** Add both flags and the summary keys. Closes when new
   `tests/test_run_smoke.py` cases with `run_row` monkeypatched (as the
   existing summary test does) show: only exit 1 reruns, serially at the
   base offset; a parallel failure that passes alone gives `passed-alone`
   and exit 1; `--from-summary` refuses each bad input with exit 2 before
   `run_row` is called; `results` is unchanged; and the existing runner
   tests pass.
2. **Docs.** The `run_smoke.py` paragraph in `docs/rules.md` Tests and the
   runner's docstring name both flags and the verdicts; the AGENTS.md
   command block gains one line. Closes when the docs-map tests and ruff
   pass.

## Suite and smoke timings (2026-10-29, DEV#25)

**Status:** `DEV#25` OPEN, recorded 2026-10-29 from the TODO audit. Stage 1
built 2026-10-30; stage 2 has its Windows local baseline, and the Linux CI
baseline waits on the next push Matt makes (see Results).

### Evidence

The CI `Test` step runs `uv run --group dev pytest -q -n auto`
(`.github/workflows/ci.yml:84-85`) and prints no per-test durations, so a
slow test is found only by its job's total. The browser job's comment
(`.github/workflows/ci.yml:87-92`) says the table "took about 15 minutes at
that width on four cores", and `summary.json` already holds each smoke's
`duration_seconds`, but it is uploaded only on failure
(`.github/workflows/ci.yml:123-125`) and nothing reads it. The last recorded
local suite time is in this plan's
[Backend type checking](#backend-type-checking) section (3,950 tests in
186.90 s, two workers, Windows/Python 3.13.15); AGENTS.md gives about 12
minutes for the Windows CI job. No Linux suite or smoke baseline is
written down.

### Design

- Add `--durations=25` to the CI `Test` step. pytest-xdist reports the
  slowest tests across workers; the step's exit is unchanged.
- `tools/smoke_timings.py <run-root-or-summary.json> [--top N]` reads one
  `summary.json` (given directly, or the single one below a run root; more
  than one is refused) and prints: the slowest N smokes with port, model,
  result and seconds; each row's total; the sum of smoke times beside the
  run's wall clock. Read-only, standard library, no server. A missing or
  malformed summary exits 2 with a message.
- The browser job gets a step after `Smokes` that runs the tool on the
  run root with `if: ${{ success() || failure() }}`, so a timing table is
  in the log whether or not a smoke failed. It cannot change the job's
  result beyond its own missing-summary exit.
- Baselines go in a dated results paragraph in this section: Linux from a
  CI run (suite top 25 and the smoke table), Windows from a local
  `pytest -n auto --durations=25` and, when main next runs it, the Windows
  CI job. Each names the commit, runner or machine and worker count.

### Non-goals

- No time budget or failure threshold on a test or smoke.
- No change to test or smoke content to make them faster in this item; a
  slow one found here becomes its own row.
- No success-path upload of the full artifact directory.

### Stages and completion criteria

1. **Tool and CI lines.** Closes when a new `tests/test_smoke_timings.py`
   checks ordering, `--top`, per-row totals and the missing, malformed and
   ambiguous summary refusals on synthetic summaries; `tests/test_ci_changes.py`
   and ruff pass; and the workflow keeps its existing steps and
   conditions apart from the two additions.
2. **Baselines.** Closes when the Linux CI log's duration and smoke tables
   and a Windows local run are recorded here with commit and worker count.
   A Windows CI figure is added when main next runs; this stage does not
   push to get one (pushes stay Matt's call).

### Results

**Stage 1, 2026-10-30.** The CI `Test` step runs
`pytest -q -n auto --durations=25`. `tools/smoke_timings.py` reads a
`summary.json`, or the only one below a directory (the CI step hands it the
`--artifacts` root, one level above the run root). It prints the slowest N
smokes with the port each ran on (worker band included), its model, result
and seconds; then each row's total, slowest first. A row is the `AGENTS.md`
row that lists the smoke, read through `run_smoke.parse_table`, so the two
7896 rows on one port stay apart. A smoke no longer in the table is its own
row, by its receipt's port and model, and is marked as such. Last comes the
sum of smoke times beside the run's wall clock (`jobs`/`offset` when the
summary has them). Only the first pass's `results` count; rerun receipts get
one line of their own. It refuses with exit 2 a missing path, a directory
with no summary or with more than one, unreadable JSON, a summary without
`results` or a numeric `duration_seconds`, and a receipt without a smoke
name and numeric duration. `--top` below 1 is refused too. The browser job's
`Smoke timings` step runs it after `Smokes` with
`if: ${{ success() || failure() }}`; the other steps and conditions are
unchanged. `tests/test_smoke_timings.py` covers it on synthetic summaries.
A real local run of smoke-flux, smoke-analysis and smoke-rewrite at
`--jobs 2` printed 105.7 s wall clock against 113.0 s of smoke time, with
smoke-analysis at 90.2 s the slowest.

**Windows local baseline, 2026-10-30** (parent 5e9c591 plus this change,
Windows 11, Python 3.13.15, AMD 16 cores / 32 threads, `-n auto` = 16
workers): 8,023 passed and 8 skipped in 176.96 s. Slowest five:
`test_fbc_runtime.py::test_actual_fbc_computes_then_reuses_and_recomputes_changed_residual`
38.07 s (setup), `test_storage_scale.py` plain 25.02 s and password
16.88 s, `test_gallery_purge.py::test_every_chunk_is_validated_before_any_requested_row_is_deleted`
16.60 s, and
`test_browser_step_containers.py::test_repository_smoke_records_reach_the_same_cold_lookup_boundary`
8.40 s. Everything else was under 8 s, and from about the 15th on under 4 s.
Non-goals file a slow one found here as its own row, but set no budget
to call one slow. Taken while Matt was away (2026-10-29), revisit if
wanted: no row is filed from this Windows run; that waits for the Linux
CI table, where the CI time goes.

**Left (stage 2):** the Linux CI baseline: the `Test` step's top 25 and the
`Smoke timings` table, from the first CI run after Matt next pushes. A
Windows CI figure follows when main next runs. Nothing was pushed to get
either. Taken while Matt was away (2026-10-29), revisit if wanted: the
local smoke table was not run in full for a Windows smoke baseline. The
plan asks only for the Windows suite, and the smokes are written on Windows
but timed in CI on Linux.

## Fake-server knob map (2026-10-29, DEV#26)

**Status:** `DEV#26` built and checked 2026-10-30 (see Results); recorded
2026-10-29 from the TODO audit.

### Evidence

`tools/fake_server.py` registers eight fake-only routes:
`/api/fake/load` (`:555`), `/step` (`:564`), `/expand` (`:580`), `/model`
(`:627`), `/lora` (`:640`), `/instruction-edit` (`:654`), `/peak` (`:661`)
and `/bin-age` (`:669`). Its parser (`:89-108`) has `--port`, `--step`,
`--data`, `--fresh`, `--model`, `--critic-delay` and `--no-critic`.
AGENTS.md's "Fake-server knobs" gotcha lists seven of the routes and omits
`/api/fake/model` (checked 2026-10-29: no occurrence in AGENTS.md).
`docs/rules.md` describes `/api/fake/model` in the model-switching tests
paragraph (`docs/rules.md:7262`) and every other route and flag somewhere
in Tests. `tests/test_docs_maps.py` holds the module layout and the smoke
table to their files but not the fake's routes or flags.

### Design

- Add `/api/fake/model` (`overrides`, `stuck_critic`, `access`) to the
  AGENTS.md knob line. Edit AGENTS.md only; CLAUDE.md just imports it.
- A new `tests/test_docs_maps.py` test reads `tools/fake_server.py` as
  text: routes from `@app_main.app.post("/api/fake/...")` and flags from
  `add_argument("--...")`. It never imports the fake, so it stays cold. It
  asserts each route and flag appears in both AGENTS.md and `docs/rules.md`,
  and that at least one route and one flag were found, so a renamed
  decorator cannot pass silently.
- The failure message names the missing item and the file it is missing
  from.

### Non-goals

- No change to the fake's routes, flags or behaviour.
- Not holding the fake's docstring, or `tests/fakes.py`, to the docs.
- Not a full description check: presence of the name only.

### Stages and completion criteria

1. **Test then line.** Closes when the new test fails on the current
   AGENTS.md naming `/api/fake/model` as missing, passes after the
   one-line AGENTS.md addition, and the rest of `tests/test_docs_maps.py`
   and ruff pass.

### Results

**Stage 1, 2026-10-30.** Recounted from `tools/fake_server.py`: the same
eight `/api/fake/...` routes and seven flags as the Evidence lists.
`tests/test_docs_maps.py::test_every_fake_route_and_flag_is_in_agents_and_rules`
reads the fake as text (routes from `@app_main.app.post("/api/fake/...")`,
flags from `add_argument("--...")`, a call split over lines included),
asserts both lists are nonempty, and matches each name as a whole token in
`AGENTS.md` and `docs/rules.md`, failing with `<name> missing from <file>`.
Written first, it failed with exactly `/api/fake/model missing from
AGENTS.md`; the AGENTS.md knob line now names `/api/fake/model`
(`overrides`, `stuck_critic`, `access`), and `docs/rules.md` Tests says what
the docs-map test holds. The docs-map file (7 tests), ruff, the pinned
Pyright gate (0 errors) and the full `pytest -n auto` (8063 passed,
19 skipped) passed. No smoke: no `web/` change.
