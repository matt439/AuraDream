# TODO audit, 2026-10-28

Matt requested an audit of the backlog and useful new features if no ready
work remained, then asked to add all proposals and fix the findings.

## Reconciliation and fixes

The starting index had 29 rows: 15 CARD, nine DEFERRED, three USER, one
DECISION and one BLOCKED. No row was OPEN; all 66 local links resolved.
The owners and source supported the main completion boundaries: performance
execution needs qualification, instruction editing stage 4 awaits review,
and credential recovery, workbench trials and accessibility need their stated
human evidence. Deferred features have not been promoted by this audit.

The following corrections are complete:

- Instruction-editing's held reason, backend comments, plan, carry summary
  and design notes now acknowledge the approved Kontext trial on 2026-10-27.
  They identify release review/integration as remaining rather than saying
  the model was never loaded or measured. The gate remains literal False,
  loading remains held and Kontext remains outside the catalogue.
- `CARD#4` and its owner now exclude the already measured Krea dev fp8/model
  load and first render. Repeatability, the other modes and checkpoints
  remain outstanding. The dated original results remain intact; the
  [fixed-loader trial](verified.md#ram-at-the-load-the-fixed-loader-2026-10-27)
  does not establish a repeated render or a cross-mode quality comparison.
- [Design requirements](design/requirements.md) no longer describe model-cache
  facts, Expand cancellation, Finish run, picked comparison/navigation or
  Show imported as missing. The concept's older labels are distinguished
  from current production capabilities. Layout validation remains `DEV#1`.
- The backup service's module summary reflects credential-backed protection,
  and the tooling plan identifies its earlier absence of a browser job as
  historical rather than current.

## Added ready work

These are requested backlog additions, with owner plans and completion
criteria; the three product features are not implemented by this audit.

| Row | Owner and first useful result |
|---|---|
| `GEN#27` | [Named generation/grid setups](plan-generation-presets.md): save several checkpoint-scoped drafts and explicitly restore one with Undo. |
| `GAL#7` | [Saved gallery searches](plan-saved-searches.md): name a query plus Starred/All models and reopen that live filter. |
| `UI#20` | [Snippet portability](plan-snippet-portability.md): export personal snippets and import through a bounded merge preview. |
| `DEV#8` | [Compiler dependency preparation](plan-regional-compilation.md#cpu-dependency-preparation): resolve optional packaging, candidate pin and notices before actual toolchain/card qualification. |

TODO's NOW starts with named setups, followed by saved searches and snippet
portability. Compiler preparation is separately visible instead of hidden
inside a CARD row. The index now has 33 rows, including four OPEN items.
These additions do not enable editing, change speed defaults, activate
backups or select a whole-app redesign.

## Validation

The existing instruction-edit gate, loader, call and queue checks passed
**91 tests**; module/smoke-map checks passed **six tests**. Repository Ruff
and `git diff --check` passed. All **212 local links** in changed/new Markdown
resolve; TODO has 33 unique row ids and exactly four OPEN rows.
Tests used the existing environment's Python
directly because the `uv` launcher failed before running tests with a
trampoline canonicalization error. No new tests were added for documentation
and copy changes. No GPU run, model download, compiler installation or live
gallery/vault operation was performed.
