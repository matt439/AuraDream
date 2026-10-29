# Review and plan: saved-library write recovery, 2026-10-28

**Status:** `UI#21` and `UI#22` are built and locally qualified. This work
follows Matt's request to
autonomously choose worthwhile new work once current scoped work is
exhausted or blocked. This is bounded UI correctness work using scratch
data. It does not change duplicate-name legality, storage formats, renderer
admission, deferred feature priorities or the gallery/credential scope.

## Ranked shortlist

| Rank | Candidate and evidence | User value, scope and risk |
|---|---|---|
| 1 | Lost successful Create replies can duplicate personal snippets and generation setups. `saveSnippet` and `saveSetup` report a failure but leave Save available after a transport/body-read error; both backend POSTs deliberately append a fresh UUID. | Prevents misleading feedback and duplicate personal data. Small UI-state change, with real fake-server write/reply-loss tests; medium regression risk around pending/selection state. Selected. |
| 2 | Snippet conflict recovery requires closing/reopening. New is allowed during the opening read, but clears `snippetsDirty`; the late list reply can then select the first saved item and undo the New action. | An in-place Reload and explicit read/edit intent make rank 1's recovery usable and protect a real existing race. Included as the necessary recovery path, without a new composer control. |
| 3 | Ordinary saved-name HTML limits count UTF-16 units while the backend counts Unicode code points. Import-copy now accepts 80 astral characters, while the ordinary name controls are capped at 80 UTF-16 units. | Definite bounded parity issue for names. Independent of write recovery; retained as the next bounded candidate, outside the first fix. |

Source evidence: `web/prompt-snippets.js` Save/Delete and opening read;
`web/generation-presets.js` Save/Delete and Reload; `backend/snippets.py`
and `backend/presets.py` new-item UUID allocation; name fields in
`web/index.html` and limits in their schema modules. The ranking uses
current local source, not competitor claims or assumed popularity.

The alternative idea of tightening malformed collection JSON was rejected:
presets and saved searches already reject recursive duplicate keys and
excessive nesting before CRUD writes. Presets accepting valid UTF-16 JSON
does not establish data loss, and rejecting it would remove compatibility.

## Write-outcome contract

An HTTP validation refusal is distinct from an unknown outcome. A network
or response-read failure, or a server 5xx, may arrive after a successful
write. Report that the result is **uncertain**, preserve the draft and hold
ordinary writes. Do not claim it was not saved/deleted and do not resend.
Known 422 refusals stay editable/retryable; 404/409 keep their stale-list
recovery behavior.

After an uncertain write, require a fresh list read and then an explicit
saved-item selection or New action. Merely reloading or reopening cannot
authorize another Create using the same retained draft. Do not guess which
duplicate-name item was created, adopt a revision silently or make a name
or payload digest an idempotency key. Existing duplicate names remain legal.
The user may deliberately select a saved result or start a new draft after
seeing the list. No server-side retry cache or API change is needed.

Snippets gain **Reload list** inside the existing dialog. It refreshes rows
while keeping editor text, target and selected old revision. Selection is
the explicit operation that reads a saved item's current revision. Track
editor/New/selection intent across asynchronous reads, including a blank
New draft whose dirty flag is false. Closing/reopening and newer requests
continue to invalidate old responses. Saved setups reuse their existing
Reload control and keep the composer unchanged.

Pending writes still hold Close/Escape and conflicting actions. Held
controls use the established aria-disabled/title/reason behavior. Snippet
insertion and setup Restore remain explicit existing operations; recovery
does not write Prompt/Negative, queue generation, switch models or change
the composer's Undo.

## Completion evidence

Add an authoritative scratch browser smoke on a FLUX.1 Krea [dev] fake:

- Let an actual snippet/setup POST commit, then lose its response. Verify
  the draft stays, the result is uncertain, and another Save sends no POST.
  Cover transport failures and 5xx results, plus a failed recovery read.
- Reload reads the committed rows, preserves the editor and still holds
  another Create. Explicit selection adopts that result's current revision;
  explicit New establishes a new draft. No name-based automatic selection.
- Unknown update/delete outcomes also require reconciliation and explicit
  intent. Known validation errors remain correctable; stale revision
  responses preserve the editor and support in-place recovery.
- Delay the opening snippet GET, press New, then release the GET. The blank
  new draft stays. Repeat with typed text and with a stale response arriving
  after Close/reopen. Dirty reloads never adopt an old selected revision.
- Check pending Close/Escape, modal Ctrl/Cmd+Enter, focus, unchanged composer,
  containment and stationary controls at standard/narrow/short viewports.
- Rerun existing snippet, generation-setup and snippet-portability smokes,
  classic-scope/docs maps, Ruff and relevant checks. No real gallery, vault,
  model, GPU run, package install or denied artifact read is involved.

## Existing-backlog audit at c44b943

This is a dated owner audit, not a reordered copy of TODO. All 31 remaining
rows were checked against their owners and relevant source, including CARD,
USER, DECISION and DEFERRED rows. Source feasibility and acceptance are
separate:

| Rows | Concrete remaining work and dependency |
|---|---|
| `CARD#1` | FLUX.1 dev/schnell fp8 inpaint at 0.95, LoRA and numerical seams: card acceptance; supported source is built. |
| `CARD#2` | FLUX.1 dev/schnell outpaint and numerical FLUX.1 seams: card acceptance; backend/dialog are built. |
| `CARD#3` | Real WSL server/dialog and dev/schnell lite inpaints: runtime/card acceptance, plus the separate local WSL access denial. |
| `CARD#4` | Remaining INT8/fp8 model/sequential combinations and Krea dev repeatability: card measurements after the fixed loader; no missing loader source found. |
| `CARD#6` | Krea 2 Turbo Windows time, Raw and klein 9B/base 9B fp8 results: card measurements; the fixed Krea dev result cannot qualify other checkpoints. |
| `CARD#7` | Dev/schnell/true-CFG refine and further offload combinations: card behavior/fit; sibling and argument support exists. |
| `CARD#9` | ERNIE two-process reproduction, 4 MP batch of two, real first-use Expand timing: runtime/card/download evidence. |
| `CARD#10` | Named fixes on the real server, typed Ctrl-C and idle loaded shutdown: real interaction/runtime qualification; fixes already landed. |
| `CARD#11` | Critic partial-placement, co-residency/rewrite pairs, critique Ctrl-C and WSL kill-on-close: real model/interaction evidence; source checks exist. |
| `CARD#12` | Official Qwen Windows fp8/sequential recheck and first bf16/sequential trial: card/load measurement. |
| `GEN#15`, `GEN#16` | FLUX.1 Krea dev peak cross-check and 1.5x Then-refine alternation/time: remaining card phases; source phases are built. |
| `PERF#1`, `PERF#3`, `PERF#4` | Compiler/cache, Native/cuDNN and FBC's finite time/memory/pixel/cancel/quality trials: private source is built; public admission stays closed pending qualification and judgment. |
| `PERF#2` | Dynamic-activation FP8 is genuinely unimplemented, but its owner explicitly waits on compilation R1; another unavailable selector would not complete an independent current slice. |
| `DEV#1` | Matt's baseline/comparison task trials and layout choice. Trial preparation is already built; another agent walkthrough is not a human result. |
| `UI#7` | Real NVDA/Narrator, touch, native notification delivery and Windows scaling; browser emulation does not supply these outcomes. |
| `GAL#6` | Intended-user native credential/session and independent-password recovery qualification; source/stubs exist, and real-vault/activation choices remain separately authorized. |
| `INP#11` | Matt reviews measured instruction drift before public catalogue/loading/replay release. Cold replay preparation is already in `cell_from_record`, edit validation, More seeds and source admission; a new planner would duplicate it. |
| `INP#5`, `INP#6` | Klein and Qwen/Nunchaku inpaint families: explicit deferrals; Qwen also needs WSL/runtime feasibility. |
| `INP#9`, `INP#10` | Crop-only inpaint and outpaint-fill/mask/fill extensions contain feasible source work but explicit later priorities; they were not silently promoted. |
| `INP#12` | Detector evidence/trial and detailer implementation require explicit commissioning; the requested plan commissions no implementation. |
| `GEN#6` | New VAE preview maps need actual map/card evidence and remain deferred. |
| `GEN#8`, `GEN#20` | External reference import and weighted/chunked SDXL prompts are technically source-feasible but explicitly deferred until Matt reopens them. |
| `PA#9` | VLM comparison scoring remains a later product/evaluation idea with unmeasured accuracy; no evaluator implementation was commissioned. |
| `UI#20` | Code and 183-step browser workflow pass; final screenshot reinspection remains after a task-artifact directory read denial. |
| `DEV#8` | Audit found optional packaging still feasible independently. It was prepared and checked offline in 81118d8. Actual wheel hash/contents/notices/RECORD and fresh online resolution remain after the denied download. |

Evidence owners: `docs/verified.md`, the inpaint/outpaint/instruction-edit and
detailer plans, `docs/plan-peak-vram.md`, `docs/plan-then-refine.md`, the
regional/attention/cache plans, `docs/review-2026-10-25.md`,
`docs/design/validation.md`, `docs/plan-automatic-backups.md` and the relevant
design-note sections. Deferred scope was checked against recorded answers,
not inferred merely from row status.

The denied screenshot action is `Get-ChildItem` on the task-local
`auradream-smoke-artifacts-4vl6i3n2` directory. The denied wheel action is
`Invoke-WebRequest` for the exact published triton-windows CPython 3.13
Windows wheel; it failed with socket permission refusal before file creation.
Neither was retried or replaced with another route. Ordinary Edge's
2026-10-27 restoration and this run's passing smokes are separate evidence.
Local WSL list/status had `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`;
later Linux browser checks were in a cloud container, not restored local WSL.

## Results

`UI#21` is implemented in `prompt-snippets.js` and
`generation-presets.js`. Unknown network/body-read or 5xx write outcomes
preserve the editor, mark the list unready and hold ordinary writes.
Fresh reads retain the uncertainty marker; explicit New/selection is the
only acknowledgement after reconciliation. Snippet Reload is in the existing
library column, and a read/edit epoch preserves New and typed intent while
an opening request is in flight. No backend or storage format changed.

The final **221-step `smoke-library-write-recovery` passed on Windows** with
fresh fake data and one worker. It lets real POST/PUT/DELETE operations
commit, then loses their replies, checks that no repeated write is sent,
and proves fresh-read plus explicit intent is needed. It covers failed
recovery reads, old revision/draft preservation, known 422 retries, real
409 conflicts, delayed opening/New/input/Close races, pending Escape/Reload,
stationary Save/Close controls and unchanged whole-composer state. It also
proves an actual pending export survives Reload while late exports from
closed/reopened dialog sessions are ignored. Export's lifecycle epoch is
separate from list-read tickets, so adding Reload cannot silently cancel
a valid in-session download.

Existing `smoke-snippets`, `smoke-generation-presets` and
`smoke-snippet-portability` all passed sequentially on independent fake
galleries. Classic-scope/docs-map checks passed **35 tests**, covering all
57 classic scripts; Ruff and diff checks pass. Independent source review
found no blocker. Six recovery screenshots from the first 160-step run were
inspected at 1400x900, 400x650 and 1400x600. The later lifecycle correction
changes request ownership, with no layout/style changes after inspection;
the final run passes containment and stationary-action assertions. No real data, weights,
vault, compiler or GPU work was involved.

## Next selected slice: Unicode saved names (`UI#22`)

Keep the backend's 80-Unicode-code-point name bound. Give only the snippet,
setup and saved-search name controls native capacity for 160 UTF-16 units
and enforce the actual 80-point limit in their save reasons. Saved searches
already count code points; snippets/setups need that explicit guard.
Do not change text/query limits, name normalization, control-character or
well-formedness rules, APIs or stored versions. This is parity, not a larger
allowed name.

Use native CDP text insertion to prove 80 astral names can be created and
edited in all three dialogs. A mixed name of 79 astral characters plus two
BMP letters has 81 points but fits 160 units; it must hold Save and send no
write. Check programmatic overlimits, unchanged 80-BMP behavior, reload/full
navigation persistence, existing library regressions and unchanged composer.
The selected implementation is a narrow HTML capacity and reason-guard
change, with no layout or runtime activation.


## Unicode saved-name result (2026-10-28)

`UI#22` is complete. The three ordinary saved-name inputs have room for
160 UTF-16 code units and retain the existing 80-code-point bound through
explicit save reasons. Native input of 80 astral characters now reaches the
backend intact. Snippet/setup guards also hold programmatically supplied
overlimits; saved searches already had that guard. The inputs describe the
80-character limit to assistive technology. Text/query limits, server bounds,
normalization, storage formats and composer behavior are unchanged.

The final **197-step** `smoke-saved-name-unicode` passed on ordinary Windows
Edge through the existing single-worker harness. It covers native astral
create/edit, a native 81-point mixed name that fits the HTML capacity but
sends no write, programmatic overlimits, 80/81 BMP bounds, item revisions,
Reload/full-navigation persistence and retained composer/empty scratch gallery.
The existing snippets, setups, saved-searches and library-write-recovery
smokes also passed: **5 of 5 in 108 seconds**. Final artifacts are beneath
`auradream-smoke-artifacts-6tpzq6dm` in this task's workspace.

An initial run passed the name behavior but exposed an overly broad fixture
write counter: changing the fixture prompt schedules the existing read-only
`POST /api/prompt/measure`. Fresh request diagnostics confirmed that call,
whose backend route measures encoders without loading an image model. The
fixture now excludes only that exact POST and setup validation from its
write count; its global route guard still refuses image generation and
unrelated mutations. No production workaround was needed. The earlier
attempt to read that failed run's temporary `fake-server.log` was denied;
that target was not retried or read by another route.

Classic-script/docs-map checks passed **35 tests**, with Ruff and diff checks
clear. The short static check was repeated once because its earlier result
was unavailable after the desktop/context interruption; no commit or browser
run was duplicated. Independent source review found no blocker. This slice
changes no styles or control positions and makes no new visual acceptance
claim for `UI#20`'s separately denied older screenshot directory. All data
was scratch; no models, GPU, credentials or real gallery were used.
