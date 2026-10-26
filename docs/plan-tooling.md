# Tooling work ready for implementation

Owner for `DEV#4`-`DEV#7`. The first three came from the
[2026-10-25 review](review-2026-10-25.md#tooling); `DEV#7` was found while
auditing the queue on 2026-10-26. The audit established this plan; dated
implementation and qualification results below record the later work.
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
are refused before launch. `LOCAL_DIFFUSION_SCREENSHOT_DIR` chooses the
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

`DEV#4` · OPEN · LOW · L. Stage 1 diagnostics are implemented; Linux
qualification and the hosted browser job remain open.

**Evidence.** `.github/workflows/ci.yml` has an always-run `lint` job and
a conditional CPU `test` job; documentation-only changes skip the latter
(changed 2026-10-26). There is no browser job. `tools/run_smoke.py` already reads the authoritative `AGENTS.md`
table, isolates each fake's gallery, offsets parallel workers' ports and
retains a failed fake's log in its temporary directory. It lacks a stable
CI artifact report. `tools/browser.mjs` knows Linux Chrome paths and uses
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

## Backend type checking

`DEV#6` · OPEN · LOW · L. The baseline stage is complete; fix bounded
batches before enabling its gate.

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
or GPU run was used. The checker is not yet in the dev group or CI;
`DEV#6` remains open for stages 2 and 3.

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
