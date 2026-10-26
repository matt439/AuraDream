# Tooling work ready for implementation

Owner for `DEV#4`-`DEV#7`. The first three came from the
[2026-10-25 review](review-2026-10-25.md#tooling); `DEV#7` was found while
auditing the queue on 2026-10-26. This is a plan, not a qualification result:
none of these changes or checks has been implemented or run by this audit.
Work in the order below. All stages use CPU, synthetic images and scratch
galleries; they need no model weights or real gallery contents.

## Portable smoke screenshots

`DEV#7` · OPEN · MED · S. Start here.

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

## Shared classic-script scope

`DEV#5` · OPEN · LOW · S. Independent of `DEV#7`.

**Evidence.** `web/index.html` currently lists 52 local classic scripts;
`app.js` is last and owns the wiring and boot. `tests/test_docs_maps.py`
checks the script count and module map, not JavaScript syntax or duplicate
bindings. The [classic-scope design contract](design-notes.md#the-page-scripts-share-one-classic-scope)
requires top-level bindings to remain shared. `app.js` itself declares
bindings, including `searchTimer`, `galleryHead` and `TOKENS_AIMED`, so
excluding it entirely would miss collisions.

**Bounded change.** Add a pytest-invoked Node check that derives the local
classic-script list from `index.html` in document order. Check every listed
script for syntax and cross-file declaration collisions, including
`app.js`, without running its boot, network requests or timers. Choose the
smallest reliable approach after checking classic-script declaration
semantics; do not maintain another script list or approximate declarations
with a regular expression. This is a load-time guard, not a DOM emulator
or a replacement for browser smokes.

**Done and checked.** The actual script list passes. Deliberate syntax
errors and duplicate lexical declarations across separate scripts fail
with useful source filenames; an `app.js` collision is covered too. Allowed
classic-script sharing still passes. Missing files and an unexpected script
kind cannot silently reduce coverage. CI provisions or verifies Node and
fails if it is missing; document any local skip policy. Run the focused
pytest, docs-map checks and an ordinary scratch-fake smoke to retain the
browser's independent answer.

## Browser smokes in CI

`DEV#4` · OPEN · LOW · L. Diagnostic preparation can start now; the final
CI job depends on `DEV#7`'s portable outputs.

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

## Backend type checking

`DEV#6` · OPEN · LOW · L. Begin with the bounded baseline stage.

**Evidence.** `backend/` uses annotations across schemas, applied requests
and storage records. `pyproject.toml`'s dev group and CI contain no type
checker, and there is no Pyright configuration. The number and nature of
initial findings are unknown; the audit did not install or run Pyright.

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
