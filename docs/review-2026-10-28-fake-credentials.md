# Review and plan: fake-server credential isolation

**Status:** `DEV#11` implemented and locally stub-qualified,
2026-10-28, after [isolated fresh data](review-2026-10-28-fake-fresh.md).

## Evidence and selection

The ranked shortlist in [setup portability](plan-setup-portability.md) identified
this gap. Importing `tests.fakes` does not run pytest's conftest/autouse
credential refusal. The fake currently imports `backend.main` without replacing
the adapter factory. A reused data directory with enabled encrypted backup
settings can start the coordinator and reach `read_secret` after due/admission
checks. Genuinely new disabled scratch data stayed cold in the earlier browser
checks; that evidence did not establish global fake credential isolation.

Source-only inspection traced the Windows adapter's lazy `_credential_api`
factory. Its read/preflight and write paths resolve that module-global factory
dynamically. It owns DLL/provider creation and CredRead/CredWrite/CredFree;
there is no delete/remove/enumeration API in this adapter. A small fake-only
process-lifetime factory refusal can therefore stop provider access without
changing production credential behavior or any OS setting.

## Bounded repair

Install a fixed RuntimeError refusal immediately after the existing PIL import
and before importing `backend.main`. Importing the credential module is cold;
do not invoke the previous factory or inspect/clear its cache. Do not add an
opt-out, fallback provider, real-vault test or credential activation. Keep the
existing adapter's fixed-error masking and coordinator failure behavior.

Only the fake tool changes; production credential/backup modules remain
unchanged. Cached status and platform facts remain factual. Manual bundle
password crypto and plaintext backups retain their existing behavior. This
guard does not isolate every effect of reused data or qualify Windows access:
keep genuinely new scratch data for browser runs and preserve `GAL#6`'s real
credential/recovery acceptance gate. No real gallery, password, vault, service,
model, browser, security configuration or deployment is used by this work.

## Qualification

Use source review and stub-only targeted CPU checks. Poison the old factory
and DLL loader, install the fake refusal without calling either, and exercise
synthetic-reference read/setup refusal without any provider call. Prove the
installer does not inspect cache/platform state. In the fake main path, stop
at the first `backend.main` import and prove refusal was installed beforehand;
never start the application or a worker for this ordering probe. Preserve
the fresh-data argument, early-PIL and mocked runner checks.

Run relevant existing credential tests under their global OS factory guard,
no-cache Ruff, pinned type checking and documentation maps. Keep source/stub
evidence separate from actual Windows credential or backup activation. Commit
only qualified owned changes on the existing local branch. The setup API/UI
proposals and unknown `GAL#12` final run stay preserved and independent.

## Implemented result and local evidence

The fake installs its fixed refusal after PIL and before `backend.main`, using
direct module binding assignment. The previous factory, cache and platform
probe are neither read nor called; no opt-out or production executable change
was introduced. Real adapter read/preflight/setup dynamically uses the refused
binding, retaining its existing fixed `CredentialError`/`credential_unusable`
behavior. Cached status is not provider qualification.

Independent source review was clear. Serial fake-tool, mocked smoke-runner,
existing credential and documentation checks passed **110 tests in 2.40 seconds**
on 2026-10-28. Four new cases bring the fake-server file to fourteen: synthetic
installation/idempotence, real adapter read/setup with synthetic references,
and main ordering stopped at the `backend.main` import boundary. Each read/
setup case entered the actual installed refusal exactly once and independently
recorded ZERO attempts at the original factory/cache/platform/DLL/error probes.
Counting matters because the adapter intentionally masks `BaseException`;
a raised sentinel alone could otherwise produce a misleading passing refusal.
No Windows credential was read, written, freed, listed or removed. The existing
adapter/credential integration checks use their supplied API/provider stubs.

No-cache Ruff initially rejected constant `setattr` (B010). Direct assignment
resolved it without an ignore, cache wrapper or production change. The final
form passed no-cache Ruff and pinned Pyright 1.1.407 over 146 backend/tools
files with zero errors/warnings; its fourteen fake-server cases passed again
in **0.32 seconds**. The existing TestClient/httpx deprecation warning remains
unrelated. The main ordering probe used synthetic modules and owned temporary
data, created no database, and never imported/started the application or worker.

This is fake-process source/stub qualification only. Actual intended-user
credential/recovery acceptance remains `GAL#6`; real service activation, card
tests and all stopped access operations remain untouched. Reused data still
has other filesystem/plaintext-backup behavior, so new scratch data remains
required for isolated browser work. `GAL#12`'s interrupted final result is
unknown; the setup API/UI proposals remain unapplied and unqualified.
