# Source-file audit, 2026-10-07

Audited the tracked Python, JavaScript, CSS and HTML in `backend/`, `web/`,
`tests/` and `tools/` at `21afcdf`. Excluded dependencies, generated assets,
review prototypes and historical documentation. Of these files, 23 exceeded
1,000 lines, including 19 application source files. Line count was a screening
tool: splits follow responsibilities, with no new build step or runtime feature.

## Implemented boundaries

| Original file | Lines before | Responsibilities after the split |
| --- | ---: | --- |
| `backend/critic.py` | 3,320 | Coordinator; `critic_runtime.py` installs/verifies the pinned runtime; `critic_server.py` owns processes and loopback requests; `critic_context.py` builds the question and budgets; `critic_reorder.py` checks pure permutations; `critic_validation.py` validates answers. |
| `backend/prompt_rules.py` | 2,422 | Rule context and dispatch; `prompt_text.py` parses text and lettering; `prompt_order.py` places tags and checks category coverage. |
| `backend/main.py` | 1,917 | App lifecycle and model/prompt/generation routes; `api_gallery.py` owns browsing, runs and deletion; `api_jobs.py` owns status, cancel, pause and SSE. |
| `web/style.css` | 3,032 | Tokens/base layout plus seven consecutive component sections in `web/styles/`, linked in the original cascade order. |
| `web/generation.js` | 2,209 | Submission/queue; `grid-preview.js`, `job-stream.js`, `job-results.js`. |
| `web/sheet.js` | 1,822 | Sheet model/axes; `sheet-render.js`, `sheet-export.js`. |
| `web/gallery.js` | 1,768 | Gallery layout/cards; `gallery-actions.js`, `reuse.js`. |
| `web/tags.js` | 1,604 | Suggestions/editing; `tag-tokens.js`, `tag-chips.js`. |
| `web/analysis.js` | 1,593 | Prompt check; `analysis-critic.js`, `comparison.js`. |
| `web/panel.js` | 1,533 | Settings/persistence; `runs.js`, `viewer.js`. |
| `web/refine.js` | 1,444 | Dialog/source/request lifecycle; `refine-prompt.js` owns prompt controls. |
| `tests/test_critic.py` | 4,696 | Runtime/install/process/server/context/tokens and answer validation, lettering, issues, reordering and sources. |
| `tests/test_prompt_rules.py` | 2,175 | Guide consistency, text/prose, lettering, order, slots and checkpoint-specific rules. |
| `tests/test_api_analysis.py` | 1,913 | Analysis routes, tags, guides, critic and lifecycle. |

`critic.py` is now 311 lines, with no critic module above 983. The stylesheet
is eight files, at most 532 lines each. The largest JavaScript file is now the
unchanged 1,070-line lightbox; the split test families top out at 800 lines.
`main.py` is 1,639 lines after extracting its independent routes.

The test families use explicit imports from small helper modules; they do not
import test modules into one another or re-collect tests through wildcard imports.
Public critic/rule entry points remain available from their existing modules.
Tests that patch implementation details target their new owning modules.

The route factories receive the app's worker and checkpoint-scope callbacks.
They read the current app-owned dependencies and do not create a second worker,
load state or settings snapshot. `main.ensure_loaded()` and the generation gate
retain their ownership and ordering.

The browser still uses classic scripts sharing one global scope, with all 26
loaded exactly once and `app.js` last. No binding was wrapped in a module or
closure. CSS links preserve the exact original rule order, including responsive
overrides and the final reduced-motion rule.

## Long files retained

| File or group | Audit decision |
| --- | --- |
| `backend/main.py` | Reduced by extracting independent routes. The remaining model/prompt/generation orchestration shares the load lock, generation gate, status and injectable services. Further extraction should introduce an explicit service context first; copying globals would break lifecycle and test substitution. |
| `backend/storage.py` (1,398) | Schema/migrations, transaction boundaries and record conversion remain together. Reporting queries are a possible later extraction with an explicit connection dependency; this pass does not alter database ownership. |
| `backend/prompt_encoding.py` (1,253) | Encoder recipes and exact template/source contracts remain beside the tokenizer cache and measurement path. A later recipe-table extraction is possible, but splitting by arbitrary line ranges would hide those contracts. |
| `backend/guides.py` (1,250), `prompt_sources.py` (1,206), `model_defaults.py` (1,135) | Mostly checkpoint-specific data. Keeping guide prose, cited analysis and profiles easy to compare is useful; size alone does not justify scattering them. |
| `backend/schemas.py` (1,195) | Related request/response contracts and normalization. A domain split is possible, but requires careful handling of shared defaults and type dependencies. |
| `backend/generate.py` (1,081) | CLI argument normalization and execution. A parser/service boundary is possible if CLI work grows; no command-line behavior changed here. |
| `web/index.html` (1,015 after adding asset links) | One page and its dialogs, kept as a single document. Splitting markup would add a template/include mechanism to the current no-build frontend. |
| `web/lightbox.js` (1,070) | One tightly coupled image-navigation/zoom/comparison state machine. Retained with its existing browser coverage. |
| `tests/test_prompt_encoding.py` (1,177) | Cohesive encoder/source-equivalence suite; preserved alongside its implementation. |
| Remaining rule dispatch and Refine lifecycle | Still around 1,000 lines, but each now has a single role. No file-size ceiling is imposed. |

## Verification

- `uv run --group dev pytest -q`: **2,114 passed, 4 skipped**, 31 dependency
  warnings, in 110.04 seconds. No GPU or weights.
- All 904 test function names remain present with the same multiplicity.
  The three split test families retain all 309 bodies/decorators and all
  1,203 collected case names/parameter IDs; normalization accounts only for
  moved module owners and the `tag-tokens.js` source-check path.
- Backend AST comparison preserved all 223 critic and 159 prompt-rule
  definitions/values after normalizing module qualifiers. The 15 extracted
  route/helper bodies also match after normalizing dependency injection and
  docstring indentation.
- Sorted OpenAPI output is identical for all 25 paths, including operation
  IDs, request/response schemas and parameter constraints.
- JavaScript executable declarations and initialization order are preserved;
  all 26 scripts pass `node --check`, including parsing the concatenation in
  page order. CSS files concatenate byte-for-byte to the original stylesheet
  after normalizing checkout line endings.
- All **34 existing browser smoke files passed**, without changing their
  expectations. Each used a fresh temporary gallery, an isolated port and a
  separate browser profile; all test servers were stopped afterward. Coverage
  includes jobs/races, cancellation, model-specific controls, refinement,
  prompt analysis, grids, sheets/export, lightbox, narrow/QHD layouts and
  Gallery beside.
- `git diff --check` passes; `AGENTS.md` and `CLAUDE.md` remain byte-identical.

These checks establish refactor equivalence at the source, API and fake-backend
levels. This pass does not add GPU quality or performance measurements to
`docs/verified.md`.
