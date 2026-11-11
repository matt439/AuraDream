# Review and repair: isolated fake-server fresh data

**Status:** `DEV#10` implemented and locally qualified, 2026-10-28.
No destructive baseline is needed: the existing command-line source directly
calls `shutil.rmtree(data)` when `--fresh` is combined with an existing data
directory. Preserve all existing data rather than exercising that operation.

## Selection and scope

The ranked follow-up shortlist in [setup portability](plan-setup-portability.md)
identified this bounded tooling issue. The codec/planner foundation is locally
qualified; the remaining API/UI proposals are preserved unapplied, and
`GAL#12` retains its unknown focused-run outcome and stopped artifact access.
This independent slice requires none of those denied actions or proposals.

Manual `tools/fake_server.py --fresh` should mean a genuinely new scratch
directory, with its path reported for inspection. Make `--fresh` and explicit
`--data` mutually exclusive before any data/environment/backend startup work.
Remove recursive deletion entirely. Keep the default shared scratch path and
explicit `--data` behavior unchanged when `--fresh` is absent. The ordinary
smoke runner already allocates each row's new `--data` and omits `--fresh`.

Keep fresh directories after shutdown for diagnostics. This changes manual
fresh invocation from deleting/reusing one directory to creating another. A
caller who supplies both flags receives an actionable argument error and can
omit `--data` for isolated fresh data or choose a different scratch directory.
Do not migrate, inspect or clean a previous data directory automatically.

This is a directory-deletion repair, not global credential or user-data
isolation for every fake invocation. Explicit/reused `--data` remains caller
chosen; enabled saved backup settings could still reach a worker. Keep the
documented requirement for genuinely new scratch before credential-cold
browser runs. No real gallery, vault, service, model or OS configuration is
activated or changed by this source work.

## Qualification

Use small standard-library argument/data helpers so validation can be tested
without starting the application. Prove both flag orders refuse before writes,
environment changes or heavy/backend imports. Under owned test temporary
roots, prove repeated `--fresh` calls create distinct empty directories and
preserve existing shared/explicit/nested sentinel files. Prove ordinary
default and `--data` calls retain their paths without deleting or creating
anything prematurely, including the runner's existing row arguments.

Run targeted CPU/source checks, no-cache Ruff, pinned type checking and
documentation checks. No browser run is required for argument/data preparation;
do not use a browser rerun or artifact recreation to close `GAL#12` or `UI#20`.
Commit only qualified owned changes on the existing local branch.

## Implemented result and local evidence

`_parse_args` now rejects either order of `--fresh` plus explicit `--data`
before selecting data, changing environment variables or importing the app.
`_data_directory` allocates a distinct retained `mkdtemp` only for `--fresh`;
the sole recursive removal and its `shutil` import are gone. Nonfresh default,
explicit, relative, missing and even previously accepted empty-string path
selection retain their behavior. Main reports the fresh directory before its
first PIL/backend import. The runner's executable behavior is unchanged; its
docstring now describes its existing new-directory `--data` behavior accurately.

Independent source review found no concrete blocker. On 2026-10-28, serial
new cold fake-server, existing mocked smoke-runner and documentation checks
passed **34 tests in 0.60 seconds**, including ten new argument/data cases.
Actual unique directories and nested sentinel files were used only beneath
owned test temporary roots. Both conflict orders stopped with exit 2 while
filesystem/environment/backend sentinels remained untouched. The fresh-main
probe stopped deliberately at its first PIL import, after checking the printed
retained path and synthetic adapter fixtures; it created no database and
started no application, model or browser. The runner's launch/cleanup tests
use mocked processes rather than actual browsers or process termination.

No-cache Ruff passed. Installed pinned Pyright 1.1.407 analyzed 146 backend/
tools files with zero errors or warnings. The existing FastAPI TestClient/
httpx deprecation warning is unrelated. No old gallery was inspected or
deleted, no vault/service/GPU/OS setting was used, and no blocked artifact or
equivalent Git search was retried. Existing `UI#20`, `DEV#8` and `GAL#12`
qualification limits and the preserved setup API/UI proposals remain separate.
