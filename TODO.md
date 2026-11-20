# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim.** Keep useful work ready for Matt's common workstation tasks. His [reported use](docs/design/preferences.md) guides priority. The candidate redesign still needs the task trials in `DEV#1`.

**Ready on 2026-11-20.** One `OPEN` row. The larger Kontext recipe, portable replay and fake one-load queue rehearsal pass. Prepare the isolated browser build and bind live guard readings to its trial receipts (`INP#13`). The [November 20 audit results](docs/maintainer/review-2026-11-20-todo.md#results-2026-11-20) record the 18 fixes and checks. Private tiled Refine has passed CPU integration and now needs the guarded card trial (`GEN#49`).

**Needs Matt.** All 13 `USER` rows are grouped here by what Matt can supply.

- Choose detailer source image IDs (`INP#12`), judge the first SDXL preview on the page (`GEN#6`), and try guarded checkpoint deletion on the real Windows cache (`GEN#72`).
- Use the [blind review pack](docs/maintainer/blind-performance-review.md) to judge the 30 fixed image pairs for Regional (`PERF#1`), Native/cuDNN (`PERF#3`) and step caching (`PERF#4`). Regional also needs a source and artifact-rebuild policy (`PERF#1`) and a shipping or Models choice (`PERF#7`). The original V6 RGB comparison failed. V7 reproduced its recovery with the retained artifacts. `PERF#2` waits for that qualification.
- Run workstation task trials (`DEV#1`), type Ctrl-C in the fake and real consoles (`DEV#23`, also covered by `CARD#10` and `CARD#11`), and check the passcode across machines (`DEV#32`). The hosted installer job requires Matt's deliberate ruleset edit (`DEV#39`).
- Check the physical phone and tablet flows (`UI#40`), and try assistive technology, touch, notifications and Windows scaling (`UI#7`).

**Card work.** Matt authorized the outstanding local GPU tests on November 18. The prompt assistant (`PA#18`) has its own explicit GPU hold. Keep the 5080's memory and host-headroom limits, and check each recipe's measured peaks in [verified.md](docs/verified.md). The three RTX 3070 rows require the remote 8 GB server.

**Decisions.** No unanswered `DECISION` rows. Matt declined [unattended backups](docs/maintainer/plan-automatic-backups.md) (`GAL#22`, `GAL#6`) and [several Then refine strengths per image](docs/maintainer/plan-then-refine.md#decision-several-strengths-per-image) (`GEN#39`). `GEN#8`, `GEN#20` and `INP#10` stay deferred.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence. Tooling row `DEV#23`: the [card harness](docs/maintainer/plan-card-harness.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#23` | USER | LOW | S+card | Six real-page assertion packs passed 2026-11-18. Left: Matt types Ctrl-C on the fake, then in the real loading/loaded consoles; temporal scrollbar coverage remains in `CARD#10`. [Remaining checks](docs/maintainer/plan-card-harness.md#real-page-follow-up-2026-11-18). |
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |

## First-run experience

The installer stages are in the [first-run review](docs/maintainer/review-2026-11-01-first-run.md#stages). The updater and launcher findings are in the [November 20 audit](docs/maintainer/review-2026-11-20-todo.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#29` | DEFERRED | LOW | S | Stage 5: hand-cut version tags for `update.cmd` to follow, and a Pinokio script; paused by priority (the installer is built). [Stage 5](docs/maintainer/review-2026-11-01-first-run.md#stages). |

## Server, security and tooling

The [peer review](docs/maintainer/review-2026-11-10-peers.md) owns the passcode and hosted installer checks. The [November 20 audit](docs/maintainer/review-2026-11-20-todo.md) owns the new tooling rows.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#32` | USER | MED | S | The opt-in passcode (`AURADREAM_PASSCODE`) is built through stage 3, 2026-11-10. Left: stage 4 on the machines, Matt's: the Linux service from a phone and another PC, Safari's autofill, a session across a restart and an update, the scripts through the passcode, a wrong passcode's lock. [Build order](docs/design-notes.md#build-order); [S2](docs/maintainer/review-2026-11-10-peers.md#s2-passcode-beyond-loopback). |
| `DEV#39` | USER | LOW | M | Local shellcheck/PSScriptAnalyzer audit is built. Left: Matt's deliberate CI ruleset edit, then a hosted installer job up to the GPU check. [T2](docs/maintainer/review-2026-11-10-peers.md#t2-the-scripts-in-ci). |


## Measurements on the card

**Owner:** [docs/verified.md](docs/verified.md): each row's "Not measured" or "Not verified" list. A result is recorded there, with its date, in the same commit that removes the row here.

| id | status | sev | size | item |
|---|---|---|---|---|
| `CARD#7` | DEFERRED | LOW | M+card | Native fp8 FLUX.1 dev Refine with offload `none` (1024-square to 1536-square, strength 0.1) is held on the 16 GB card by the weights-only guard bound. Only if Matt later selects this mode on viable hardware: freeze header/activation/host budgets before a load. No wider family/offload sweep is commissioned. [Owner](docs/verified.md#windows-follow-up-measurements-2026-11-18). |
| `CARD#9` | DEFERRED | LOW | S+card | Revisit ERNIE fp8 at 2048-square batch two when a viable hardware or recipe envelope exists. The projected 17,587 MiB exceeds the 15,200 MiB ceiling. [Measured rungs](docs/verified.md#ernie-reproduction-and-actual-first-use-expand). |
| `CARD#10` | CARD | LOW | M+card | Verify actual scrollbar change on the real server after the `DEV#76` fixture, and hand-type Ctrl-C during loading and while idle with a loaded model after the fake-console prerequisite. [Remaining checks](docs/maintainer/plan-card-harness.md#real-page-follow-up-2026-11-18). |
| `CARD#11` | CARD | LOW | M+card | Check a pure reorder that reaches partial placement, repeat real WSL owner-death cleanup after `DEV#72`, and hand-type Ctrl-C during a critique. [Placement](docs/verified.md#critic-partial-placement-trigger-follow-up-2026-11-18) · [WSL result](docs/verified.md#wsl-critic-after-abrupt-owning-app-death-2026-11-18). |
| `CARD#14` | CARD | MED | S+card | On the 3070 server: a suspend with a model loaded, and whether the loaded pipeline survives the resume (`NVreg_PreserveVideoMemoryAllocations=1` is set). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#15` | CARD | LOW | S+card | Nunchaku int4 on the 3070 (sm_86): whether the lite-infer kernels load and render, its time and peak. [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#16` | CARD | LOW | S+card | The prompt critic's pinned CUDA 13.4 llama.cpp build under the 3070's 13.2 driver (first use, a critique). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#21` | DEFERRED | LOW | M+card | FLUX 3 on the 8 GB 3070 as lite-infer int4 (its index points at nvfp4, which sm_86 cannot run): wire the int4 load, then fit and time; the upsampler cannot fit. Only if Matt wants FLUX 3 there. [FLUX 3](docs/design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders). |

## Inpainting, outpainting and edits

**Owner:** [docs/maintainer/plan-inpainting.md](docs/maintainer/plan-inpainting.md#phases) and [docs/maintainer/plan-outpainting.md](docs/maintainer/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting). `INP#11` and `INP#12`: the [instruction-editing plan](docs/maintainer/plan-instruction-editing.md) and [detailer plan](docs/maintainer/plan-detailer.md), requested 2026-10-27.

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. [Phase 4](docs/maintainer/plan-inpainting.md#phases). |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. [Phase 5](docs/maintainer/plan-inpainting.md#phases). |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. [Plan](docs/maintainer/plan-outpainting.md#phases). |
| `INP#12` | USER | LOW | L+card | Matt chooses source image IDs for the authorized small-region quality, identity, seam and memory comparison. Illustrated detection has low recall and illustrated hands remain unqualified. [Card acceptance](docs/maintainer/plan-detailer.md#remaining-card-acceptance). |
| `INP#13` | OPEN | LOW | L+card | Build the isolated Kontext browser trial and bind live guard readings to its load, edit and unload receipts, then qualify the frozen aspects and 1024-square recipe. [Trial](docs/maintainer/plan-instruction-editing.md#second-kontext-recipe-and-trial-inp13). |

## Generation, jobs and models

**Owner:** the plan or section each row links; `GEN#48`-`GEN#56`: the [2026-11-10 peer review](docs/maintainer/review-2026-11-10-peers.md)'s R1-R10. Otherwise: [peak memory](docs/maintainer/plan-peak-vram.md) and [Then refine](docs/maintainer/plan-then-refine.md) for the card rows; the design notes, [docs/verified.md](docs/verified.md) or a review for the deferred ones.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#16` | CARD | LOW | S+card | A guarded Krea 1.5x alternation and pair-time comparison still needs viable card/host headroom: matched-stage preflight refused before loading on 2026-11-18. No pass/timing qualified. [Result and next admission](docs/maintainer/plan-then-refine.md#matched-stage-preflight-2026-11-18). |
| `GEN#6` | USER | LOW | S | The guarded Animagine card capture passed 2026-11-17, with identical callback/SSE preview bytes and retained first/middle/late/final images. Left: Matt's real-page glance and preference for the noisy first frame. FLUX.2's map stays deferred. [Evidence](docs/maintainer/review-2026-11-12-todo.md#gen6-card-capture-2026-11-17). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#20` | DEFERRED | LOW | L+card | Opt-in A1111 weighting and 77-token chunking on SDXL, deferred by Matt 2026-10-27 because he does not write weighted prompts; revisit on request. [10-24 G10](docs/maintainer/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |
| `GEN#49` | CARD | LOW | L+card | Qualify the private tiled Refine adapter on the 5080 for fidelity, repeat and cancellation recovery, then qualify the public flows. CPU adapter and durable replay passed. [Stage 3](docs/maintainer/plan-tiled-refine.md#3-guarded-private-card-qualification). |
| `GEN#72` | USER | LOW | L | A guarded delete in Models > On disk, any whole checkpoint, built through stage 4 2026-11-14 (shared repositories file by file). Left: Matt's acceptance on his real Windows cache: one entry with another selected (measured vs planned bytes, Models says not downloaded, the selected row held), then one Anima variant with the other two still loading; result in `docs/verified.md`. [Plan](docs/maintainer/plan-model-delete.md#stages-and-completion-criteria). |

## Speed

Each row links to its current measurement or dependency. The [original speed review](docs/maintainer/review-2026-10-25.md#speed) records how the candidates were chosen.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#1` | USER | MED | L+card | Bounded Regional table, early-request latency and artifact-local recovery/control replay measured 2026-11-18. V6's historical RGB gate remains failed; v7 reproduces its recovery twice with retained artifacts. Left: Matt's pair judgment and source/release/artifact-rebuild policy. Public admission stays closed. [Owner](docs/maintainer/plan-regional-compilation.md#retained-artifact-uncancelled-control-v7-2026-11-18). |
| `PERF#3` | USER | LOW | M+card | Fixed Native/cuDNN speed, fit and exact replay comparisons plus the separate capped-resident-v3 same-process lifecycle were measured 2026-11-18. Left: Matt judges visual usefulness/quality before release decisions. Original fresh-headroom refusal is retained; ordinary uncapped/UI switching and public nonempty admission remain unqualified/closed. [Results](docs/maintainer/plan-attention-backends.md#private-card-results-2026-11-18). |
| `PERF#4` | USER | LOW | L+card | Six Off/FBC pairs, fresh-process replay, state cleanup and cancel/recovery passed 2026-11-18. Left: Matt judges prompt coverage, anatomy, detail and text in the fixed pairs before any routine-use/release decision. Public caching remains closed. [Results and pairs](docs/maintainer/plan-step-caching.md#private-card-results-2026-11-18). |
| `PERF#2` | BLOCKED | LOW | M+card | `fp8-dynamic` (PerTensor vs fp8) waits on `PERF#1`'s acceptance/qualification decision. Its own stage 2 loader mapping, inventory, refusals and reproduction carry are unbuilt and precede its independent activation-kernel/card qualification. Existing fp8 measurements do not measure dynamic activations. [Dependency and stages](docs/maintainer/plan-fp8-activations.md#current-dependency-2026-11-18). |
| `PERF#7` | USER | LOW | — | The first Regional card table is measured. Matt now judges whether it should ship and where: an opt-in Models choice with Eager default, launch-only, or held. Cache location, size policy and static shapes are decided. `PERF#1`'s remaining qualification and public-release gates stay separate. [Questions](docs/maintainer/plan-regional-compilation.md#matts-questions). |

## Prompt analyser

**Owner:** the design notes' [prompt analyser](docs/design-notes.md#the-prompt-analyser) for the deferred item; its [open questions](docs/design-notes.md#open-questions-from-the-analyser) for decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. [Prompt analyser](docs/design-notes.md#the-prompt-analyser). |
| `PA#18` | CARD | MED | M+card | The assistant conversation and draft tools are built. The remaining GPU checks cover latency, memory, process cleanup and prompt quality. Matt explicitly holds these runs. [Plan and test handoff](docs/maintainer/plan-prompt-assistant.md). |

## The page

**Owner:** [docs/design/validation.md](docs/design/validation.md) for the manual checks; [the phone review](docs/maintainer/review-2026-11-05-phone.md) for the phone redesign.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#40` | USER | MED | M | Phone acceptance: physical iPhone keyboard/sharing/gestures, Android/system and native iOS edge Back, landscape instruction caret, actual iPad Split View, and Notify when done delivered through the service worker on a real Android phone and an iPhone home-screen app (`UI#55`). [Phone review](docs/maintainer/review-2026-11-05-phone.md#10-edit-dialog-keyboard-acceptance-2026-11-07). |
| `UI#7` | USER | LOW | M | Hands-on NVDA/Narrator, touch, native notification delivery and actual Windows scaling checks; browser emulation and accessibility-tree inspection do not complete them. [Manual checks](docs/design/validation.md#accessibility-and-desktop-checks). |

## Review coverage

Each review's remaining work is indexed above; closed findings stay in their owners.

| review | disposition |
|---|---|
| 2026-08-29, 2026-09-07 | Every lead re-checked in 2026-09-07's findings; its checklist is all ticked. |
| 2026-09-28 | Its checklist; its residues were built 2026-10-17 (`PA#1`-`PA#4`). F47, the CLI's literal `{word}` and the other entries under "Accepted, and worded as trade-offs" are decided, not open. |
| 2026-10-03 | All confirmed findings and P1 fixed 2026-10-04. |
| 2026-10-04 UI | All done but U61 B (`GEN#8`); U5, U12, U79, U100 and U108 were left by decision. |
| 2026-10-05 areas | 84 fixed, A68 partly by decision; the integration review's K1-K5 fixed. |
| 2026-10-12 | R1-R53 fixed; F1-F5, F7-F13 and F15-F19 built. F6's supported inpaint/outpaint paths are built; remaining families/measurements are `INP`/`CARD`. F14 was qualified on the card 2026-10-21; F5's age-scoped emptying was built 2026-10-19 (`GAL#5`). |
| 2026-10-24 | G1 smoke runner and G2 lint/CI built the same day; G5 choices, G6 snippets, G7 queue, G8 size/crop and G9 help built 2026-10-25/26 ([choices](docs/maintainer/plan-random-prompt-variables.md), [snippets](docs/maintainer/plan-prompt-snippets.md), [queue](docs/maintainer/plan-queue-controls.md), [size/crop](docs/maintainer/plan-size-axis-crop-anchor.md), [help](docs/maintainer/plan-shortcuts.md)); G3, G4 and G10 indexed above; the rest recorded as not taken. |
| 2026-10-25 | R1-R10 indexed above. Tooling has [execution stages](docs/maintainer/plan-tooling.md); speed/backups have CPU-only planning starts. Matt requested both edit plans 2026-10-27. `INP#11` passed bounded card acceptance on 2026-11-18. `INP#12` now awaits his source IDs. |
| 2026-10-26 workflow | Three reproduced findings fixed and qualified: mixed prompt prose (`PA#10`), incomplete numbers (`GEN#24`) and unseen queued outcomes (`GEN#25`). [Evidence and results](docs/maintainer/review-2026-10-26-ux.md). |
| 2026-10-26 keyboard | All three approved stages (`UI#17`-`UI#19`) fixed and qualified, including fast routes and compact setup. [Evidence](docs/maintainer/plan-shortcut-workflow.md). |
| 2026-10-26 sizing/Bundles | Password association (`UI#14`), clear-count explanation (`UI#15`) and frequent secondary target sizing (`UI#16`) fixed and qualified. [Evidence and results](docs/maintainer/review-2026-10-26-sizing-bundles.md). |
| 2026-10-26 TODO audit | [Owner/code reconciliation](docs/maintainer/review-2026-10-26-todo.md): Marian chunk truncation (`GEN#26`), portable screenshots (`DEV#7`), restored critic decision (`PA#11`) and bounded tooling/planning stages. |
| 2026-10-26 overnight | `GEN#26`, `DEV#5`-`DEV#7` built and locally qualified; all four feature plans written. [Tooling evidence](docs/maintainer/plan-tooling.md). Linux browser qualification (`DEV#4`) closed with PR #15 (hosted, 77 of 77); no GPU acceptance is claimed. |
| 2026-10-27 decisions | Matt deferred `GEN#20`; the requested [instruction-editing](docs/maintainer/plan-instruction-editing.md) and [detailer](docs/maintainer/plan-detailer.md) plans are written. Matt then commissioned starting `INP#11` without the card; `INP#12` was deferred at the time and commissioned on 2026-11-16. `PA#11` closed by keeping critic rewrites available with the existing [omission warning](docs/design-notes.md#open-questions-from-the-analyser). |
| 2026-10-28 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-10-28-todo.md): corrected stale edit/design status and `CARD#4` scope; added the requested named setups, saved searches, snippet portability and separate compiler preparation. Named setups (`GEN#27`, [plan](docs/maintainer/plan-generation-presets.md#browser-acceptance-2026-10-28)) and saved gallery searches (`GAL#7`, [plan](docs/maintainer/plan-saved-searches.md#browser-acceptance)) are built and locally qualified. |
| 2026-10-28 PNG | Named [JSON decoder refusals (`UI#26`)](docs/maintainer/review-2026-10-28-png-json-refusals.md) and [unreadable-size fallback (`UI#27`)](docs/maintainer/review-2026-10-28-png-size-refusals.md). |
| 2026-10-28 library | Recovery owners: [snippet/setup libraries and remaining owner audit](docs/maintainer/review-2026-10-28-library-recovery.md), [setup list recovery (`UI#23`)](docs/maintainer/review-2026-10-28-setup-list-recovery.md), [saved searches (`UI#24`)](docs/maintainer/review-2026-10-28-saved-search-recovery.md). |
| 2026-10-28 gallery | `GAL#8`-`GAL#11`, `GAL#13` and `GAL#16` (picked favorites, favorite write order, exact and broad bin order, local date bounds, numeric scalars) are owned by their `docs/maintainer/review-2026-10-28-*.md`; the search slices by their plans: [date ranges (`GAL#14`)](docs/maintainer/plan-gallery-date-ranges.md), [orientation (`GAL#15`)](docs/maintainer/plan-gallery-orientation.md), [area (`GAL#17`)](docs/maintainer/plan-gallery-area-search.md). |
| 2026-10-28 jobs | Owners: [queued cancellation receipts (`GEN#28`)](docs/maintainer/review-2026-10-28-queued-cancel-recovery.md), [watched action replies (`GEN#29`)](docs/maintainer/review-2026-10-28-watched-action-replies.md), [wrapped seed headlines (`GEN#30`)](docs/maintainer/review-2026-10-28-more-seeds-headline.md). |
| 2026-10-28 tooling | `DEV#9`-`DEV#19` built, each owned by its `docs/maintainer/review-2026-10-28-*.md`: fatal prerequisites, fresh fake directories, fake credentials, smoke port plan, fake CLI numbers, viewport and browser port preflights, step containers, preparation order, startup viewport and input refusals. [First of them](docs/maintainer/review-2026-10-28-fatal-prerequisites.md). |
| 2026-10-29 gallery | [Runs scope and deliberate retry (`GAL#18`)](docs/maintainer/review-2026-10-28-runs-scope-recovery.md), including actual-handler/native controls and new screenshot inspection. |
| 2026-10-29 grid | [Closed preview ownership (`UI#29`)](docs/maintainer/review-2026-10-28-grid-preview-close.md), with actual-module/native controls and scoped screenshot inspection. |
| 2026-10-29 copy feedback | [Original image and inspect-session ownership (`UI#30`)](docs/maintainer/review-2026-10-29-copy-feedback-ownership.md), with synthetic-clipboard native controls and two inspected new screenshots. |
| 2026-10-29 API paging | [SQLite-bound gallery/bin queries (`GAL#20`)](docs/maintainer/review-2026-10-29-gallery-page-bounds.md), with real overflow controls, legal-MAX binding and existing paging/scope regressions. |
| 2026-10-29 Check dismissal | [Retained findings and dismissed-panel focus (`UI#31`)](docs/maintainer/review-2026-10-29-check-dismissal.md), with actual-module/original controls, strict fake native checks and three inspected screenshots. |
| 2026-10-29 pending slices | Qualified with negative controls, an independent review and inspected screenshots: [Bundle & clear (`GAL#12`)](docs/maintainer/review-2026-10-28-bundle-clear-order.md), [Reuse-pass opening (`UI#28`)](docs/maintainer/review-2026-10-28-reuse-pass-opening.md), [split extra (`GEN#31`)](docs/maintainer/review-2026-10-29-split-extra-queue.md); screenshots then closed [`GAL#19`](docs/maintainer/review-2026-10-29-runs-favorite-counts.md) and [`UI#20`](docs/maintainer/plan-snippet-portability.md). |
| 2026-10-29 setup portability | [Saved-setup export/import (`UI#25`)](docs/maintainer/plan-setup-portability.md#stage-3-implementation-and-browser-evidence-2026-10-29): cold codec, atomic API and the browser merge dialog, with an owned-scratch smoke and three inspected new screenshots. |
| 2026-10-29 compiler dependency | `DEV#8` closed: the exact triton-windows wheel's hash, metadata, RECORD and bundled NVIDIA/TinyCC notices were inspected, and a fresh online resolution reproduces the lock. [Results](docs/maintainer/plan-regional-compilation.md#cpu-dependency-preparation). |
| 2026-10-29 CI browser startup | [Bounded browser startup](docs/maintainer/review-2026-10-29-ci-browser-startup.md): hosted qualification passed on e687fcd and on main after PR #16. |
| 2026-10-30 overnight | Built and merged in PR #22: `GEN#36`-`GEN#38`, `GAL#23` (the Gallery section emptied), `DEV#26`, `DEV#21` and stage 1 of `DEV#20`/`DEV#23`, then an integration review and load-flake fixes (polls for fixed sleeps, a bundle settle's hold, profile cleanup, sidecar waits). [Owners](docs/maintainer/review-2026-10-29-backlog.md); [card harness](docs/maintainer/plan-card-harness.md). |
| 2026-10-29 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-10-29-todo.md): stale owner text fixed; 27 OPEN rows and four new DECISION rows added at Matt's request. |
| 2026-10-31 code hygiene | Stale docs, dead code, duplicates and long files on `chore/code-hygiene`; no row opened. The splits and retained long files are [source-audit.md's third pass](docs/maintainer/source-audit.md#third-pass-2026-10-31). |
| 2026-11-01 first run | [First-time setup and an installer](docs/maintainer/review-2026-11-01-first-run.md): `DEV#27`-`DEV#29`, `UI#39`, `GEN#40`, `GEN#41`; the token wizard, release ZIP, user-wide data relocation and signed/bundled apps recorded as not taken. |
| 2026-11-03 personas | [Personas and stories against the app](docs/maintainer/review-2026-11-03-personas.md): design docs and usage.md reconciled; rerun-preview notes, cross-tab Stopping…, scope titles and held Orientation fixed; its six questions answered 2026-11-03 and the placements built (`DEV#30`, closed). |
| 2026-11-05 phone | [Phone redesign](docs/maintainer/review-2026-11-05-phone.md#9-remaining-implementation-findings-2026-11-07): all implementation phases and remaining code fixes built; browser and native image-pass keyboard evidence retained. Physical/platform acceptance remains `UI#40` (`USER`). |
| 2026-11-10 peers | [The app against comparable projects](docs/maintainer/review-2026-11-10-peers.md): 48 candidates checked against the code and earlier decisions, 23 kept (`GEN#48`-`GEN#56`, `INP#13`, `DEV#31`-`DEV#40`, `UI#41`-`UI#43`), the rest recorded as not taken; `DEV#31`, `DEV#32` (through stage 3), `DEV#34`-`DEV#38`, `DEV#40`, `GEN#51`-`GEN#54`, `GEN#56`, `UI#42` and `GEN#57` (opened and built the same day) built 2026-11-10; `GEN#50`, `GEN#55`'s first slice, `DEV#33`, `UI#41`, `UI#43` and `DEV#39`'s local audit built 2026-11-11; `GEN#49`, `INP#13`, `DEV#32` stage 4 and `DEV#39`'s hosted job indexed above; `GEN#48` completed its bounded Anima release on 2026-11-17. |
| 2026-11-10 first time | [A first-time user, after the installer](docs/maintainer/review-2026-11-10-first-time.md): 56 findings checked by skeptics, one refuted; merged to 28, 27 as rows (`DEV#41`-`DEV#53`, `GEN#58`-`GEN#62`, `UI#44`-`UI#50`, `PA#13`, `PA#14`), all built 2026-11-10/11; F27 closed with `GEN#65`'s shard maps (2026-11-12). |
| 2026-11-12 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-11-12-todo.md): stale NOW and coverage text fixed, `DEV#20` stage 2 recorded; 26 verified proposals added as 14 `OPEN` rows and 13 `DECISION` rows; Matt accepted every recommendation the same day (11 more `OPEN`, `CARD#4`/`CARD#6`/`CARD#12` retired). Built 2026-11-12/13 with `GAL#27` and `UI#53` (PR #71, [results](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12)), `DEV#58` last (PR #72, every smoke polling its expects); what is left of `GEN#6` and `INP#12` is indexed above; `GEN#48` completed its bounded Anima release on 2026-11-17; `GEN#66` closed 2026-11-14 on Matt's `powercfg /requests` reading, `GEN#68` the same day on his look at On disk (its delete is `GEN#72`). `GEN#46` closed 2026-11-13 (PR #73): CI's account reads both FLUX 3 licences, and FLUX 3 and Klein Base are `REAL` in the encoding tests and the CI tokenizer fetch. |
| 2026-11-13 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-11-13-todo.md): coverage table, `GEN#15`'s owner and two F27 references fixed; two rounds of lenses, 27 proposals, 4 refuted; 16 `OPEN` rows and 2 `DECISION` rows added, both accepted by Matt the same day (`GAL#31`, `UI#56`, `OPEN`); `DEV#64`, `GAL#28`, `GAL#29`, `GEN#71`, `GAL#30`, `GEN#69`, `GEN#70`, `PA#17`, `UI#54`, `UI#55`, `DEV#65`, `DEV#66`, `DEV#67`, `DEV#68`, `DEV#69`, `DEV#70`, `GAL#31` and `UI#56` built the same day ([results](docs/maintainer/review-2026-11-13-todo.md#results)). |
| 2026-11-16 detailer | `INP#12` stages 1-3 built: experimental CPU face/hand finders, region review, serial SDXL inpaints, portable provenance and partial-result recovery. Local CPU/browser checks pass; separate card acceptance remains. [Evidence and behavior](docs/maintainer/plan-detailer.md#implementation-2026-11-16). |
| 2026-11-18 local GPU qualification | Completed `CARD#1`, `CARD#2`, `CARD#3`, `CARD#13` and `INP#11`; remaining unsafe/manual/coverage cases stay explicit. Private Regional, Native/cuDNN and FBC matrices are measured; pixel-changing pairs await Matt. WSL critic owner death exposed a production lifetime gap. [Results](docs/verified.md). |
| 2026-11-20 TODO audit | [Reconciliation and 18 verified fixes](docs/maintainer/review-2026-11-20-todo.md#results-2026-11-20). The detailer comparison awaits Matt's source IDs and the above-budget ERNIE case awaits a viable envelope. |


---

## How to read this file

Rules for writing it: read them before editing a row, not before picking one.

**This file is an index, not a work order.** One row per open item: an id, a status, two grades, one clause and a pointer. Why an item matters, how to build it and what counts as done belong in its owner (a plan, the design notes, a review, `docs/verified.md`); a second copy here goes stale unnoticed.

1. **A row leaves this file in the same commit that finishes it.** Record the result in the owner (a measurement in `docs/verified.md` with its date, a build in the plan's status or the design notes) and delete the row; no closed rows stay here. An emptied section becomes one line under "Review coverage" or is dropped.
2. **The owner is the source of truth.** When this file and its owner disagree, this file is stale.
3. **When part of an item lands, replace the clause with what's left;** don't append to it. Keep a row under about 500 bytes.
4. **New work starts in an owner:** a plan in `docs/maintainer/plan-*.md` for a feature, a review's findings for a defect, the design notes for a deliberate gap. Then it gets a row here.
5. **NOW changes only when its items finish or Matt reorders them.** Refresh its ready count when rows or statuses change.

**`id`:** a stable handle, not an order; a retired id is never reused. The prefix is the section: `CARD` measurements on the card, `INP` inpainting, outpainting and edits, `GEN` generation, jobs and models, `PERF` speed, `GAL` gallery and storage, `PA` prompt analyser, `UI` the page, `DEV` docs and tooling. A review's own id (10-12 F7, U61) is named in the clause. One id was reused all the same: `GEN#23` named a wording fix (opened and closed 2026-10-25), then [More seeds' redesign](docs/maintainer/plan-more-seeds-ux.md) (2026-10-26, closed with `UI#13`).

**`status`**, a closed list:

- `OPEN`: implementation or CPU/browser checks can start; the owner defines completion. `+card` still requires a separately authorized card pass before closure.
- `CARD`: the next step is a run on the RTX 5080 (or, where the clause says so, the 8 GB RTX 3070 Linux server). Matt runs it or OKs it; one size at a time, never past the card.
- `USER`: the next step needs Matt's hands-on trial or manual validation; listed in NOW, separately from ready work.
- `DECISION`: the next step is Matt's call. Always listed in NOW.
- `BLOCKED`: waits on another row (named in the clause).
- `DEFERRED`: paused by priority; picked up only when NOW admits it or its condition is met.

**`sev`:** the cost of leaving it open, not difficulty. `HIGH` a built feature with no measurement behind it, or a decision that holds up other work; `MED` a real defect or gap, but bounded; `LOW` convenience, polish, doc debt.

**`size`:** engineer-hours: `S` up to 2 h, `M` 2-6 h, `L` 6-20 h, `XL` over 20 h. `+card` means the real cost is time on the card, which Matt runs or OKs.

**Decided against:** what the docs record as a trade-off or a decision not to build (a review's "Accepted" or "Kept by decision") never gets a row.
