# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-18):** keep useful work ready for Matt's common workstation tasks. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog (updated 2026-10-28):** two `OPEN` rows remain from Matt's request after the [TODO audit](docs/review-2026-10-28-todo.md). Next snippet export/import (`UI#20`); its owner plan defines bounded implementation and CPU/browser acceptance. `DEV#8` separates compiler dependency/lockfile/notices preparation from `PERF#1`'s toolchain/card qualification. Saved gallery searches (`GAL#7`) are built and locally qualified ([plan](docs/plan-saved-searches.md#browser-acceptance)), as are named generation/grid setups (`GEN#27`, [plan](docs/plan-generation-presets.md#browser-acceptance-2026-10-28)). `DEV#4` is complete: PR #15's hosted browser job passed all 77 smokes on main ([tooling](docs/plan-tooling.md#browser-smokes-in-ci)).

**Performance stages:** CPU dependency preparation is `DEV#8`, then `PERF#1` ([compilation](docs/plan-regional-compilation.md)), `PERF#3` ([attention](docs/plan-attention-backends.md)) and `PERF#4` ([caching](docs/plan-step-caching.md)). Performance source, bounded diagnostics and Windows browser checks are complete with Eager defaults and closed execution guards; real toolchain/card qualification and nonempty release admission remain. Default-off backups, explicit shared mode/retention and credential source/stubs are built; actual credential/recovery qualification remains. No real gallery was activated. `PERF#2` still waits on `PERF#1`; source implementation is separate from the remaining layout trials.

**Needs Matt:** a review of `INP#11` stage 3's Kontext numbers before stage 4 opens the gate; workbench task trials (`DEV#1`), hands-on accessibility/desktop checks (`UI#7`) and intended-user Windows credential/recovery qualification (`GAL#6`, source/stubs built; no real vault accessed). The four decisions were answered 2026-10-27: SDXL weighting is deferred because Matt does not write weighted prompts (`GEN#20`); the requested instruction-editing and detailer plans are written, with `INP#11` now started and `INP#12` implementation unscheduled; critic rewrites stay available with the omission warning (`PA#11`, closed in its owner). Whole-app layout implementation waits for its trials; it remains a design hypothesis.

**Card prerequisite:** done 2026-10-27 (`GEN#22`): the fixed loader's cold fp8 FLUX.1 Krea dev load peaked at 27.9 GiB working set, down from 46+ ([verified](docs/verified.md#ram-at-the-load-the-fixed-loader-2026-10-27)). Each other checkpoint still needs its own measured load and render.

**GPU runs are Matt's to run or OK, never past the 16 GB card** (every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

Matt's approved [shortcut and compact-setup stages](docs/plan-shortcut-workflow.md) and earlier workflow/Bundles fixes remain complete. Speed defaults and unattended backup policy remain unsettled in their owners. `INP#11` stages 0-3 are done; stage 4 waits on Matt's review of stage 3's numbers. The detailer plan does not commission implementation. Other deferred features remain deferred.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |

## Measurements on the card

**Owner:** [docs/verified.md](docs/verified.md): each row's "Not measured" or "Not verified" list. A result is recorded there, with its date, in the same commit that removes the row here.

| id | status | sev | size | item |
|---|---|---|---|---|
| `CARD#1` | CARD | LOW | S+card | Inpaint at fp8: FLUX.1 dev and schnell at 0.95, a LoRA through a FLUX.1 inpaint, the seam by number on FLUX.1. Nunchaku is `CARD#3`, later families are `INP`. [Inpaint](docs/verified.md#inpaint). |
| `CARD#2` | CARD | LOW | S+card | Outpaint: FLUX.1 dev and schnell, and the outpaint's seam by number on FLUX.1. [Outpaint](docs/verified.md#outpaint). |
| `CARD#3` | CARD | LOW | S+card | An inpaint under WSL through the real server and the Inpaint dialog, and over the FLUX.1 dev and schnell lite transformers. [Under Nunchaku](docs/verified.md#under-nunchaku-wsl-2026-10-25). |
| `CARD#4` | CARD | LOW | M+card | INT8/fp8 with model/sequential offload on FLUX.1 Krea dev, dev, schnell, Krea 2 Turbo/Raw and klein 9B/base 9B, except Krea dev's measured fp8/model load and first render (repeatability remains). INT8 sequential on other checkpoints. Qwen is `CARD#12`. [Remaining modes](docs/verified.md#quantization-and-offload-2026-10-25). |
| `CARD#6` | CARD | LOW | M+card | Windows fp8 rows: Krea 2 Turbo's time, Krea 2 Raw, FLUX.2 [klein] 9B and base 9B. Raw and base 9B were not tried. [RAM at the load](docs/verified.md#ram-at-the-load-2026-10-25). |
| `CARD#7` | CARD | LOW | M+card | Refine on FLUX.1 at fp8: dev, what schnell's starting strength keeps, true CFG through FluxImg2Img. Offload none and sequential past Z-Image Turbo. [Refine pass](docs/verified.md#refine-pass). |
| `CARD#9` | CARD | LOW | S+card | ERNIE-Image Turbo on fp8: seed reproduction across two processes, whether the 4 MP batch of two fits (the harness's projection refused it), and the page's Expand line and clock over the rewriter's real first-use download. [Behaviour](docs/verified.md#behaviour). |
| `CARD#10` | CARD | LOW | M+card | On the real server, fixes checked only on the fake: 9377de1, 7733f81, 6a7570d, ad01fdc, 2a4d2cc, 2b89ed5. R50's fix (1744dc3) with a typed Ctrl-C, and Ctrl-C to an idle server with a model loaded. [Behaviour](docs/verified.md#behaviour); [10-12 not reached](docs/review-2026-10-12.md#review--2026-10-12). |
| `CARD#11` | CARD | LOW | M+card | The critic: a reorder that reaches the partial-placement check (none of 16 did; every 4B reorder changed words); FLUX.1 Krea [dev] at int8 or sequential beside it, and its rewrites against originals over more seeds; kill-on-close under WSL; Ctrl-C during a critique. [Prompt analyser](docs/verified.md#prompt-analyser). |
| `CARD#12` | CARD | LOW | S+card | Official Qwen-Image on Windows: recheck fp8 with sequential offload; BF16 with sequential offload is untried. [RAM at the load](docs/verified.md#ram-at-the-load-2026-10-25); [Switching](docs/design-notes.md#switching-checkpoints-while-the-server-stays-open). |

## Inpainting, outpainting and edits

**Owner:** [docs/plan-inpainting.md](docs/plan-inpainting.md#phases) and [docs/plan-outpainting.md](docs/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting). `INP#11` and `INP#12`: the [instruction-editing plan](docs/plan-instruction-editing.md) and [detailer plan](docs/plan-detailer.md), requested 2026-10-27.

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#9` | DEFERRED | LOW | M | Phase 6, the masked area only (`padding_mask_crop` at a working size, a recorded `mask_crop`); not on Z-Image, whose class lacks it. Also for large SDXL outpaint canvases. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |
| `INP#11` | DECISION | LOW | XL+card | Stages 0-3 done; stage 3's Kontext trial (fp8, model offload, 512x512) fits at 14.5 GB, replays, cancels and unloads, but its edits drift beyond the instruction ([verified](docs/verified.md#instruction-edit)). Next: Matt reviews those numbers, then stage 4's gate, catalogue entry and Re-render/More seeds of edits. [Plan](docs/plan-instruction-editing.md#stages-and-completion-criteria). |
| `INP#12` | DEFERRED | LOW | L+card | Face/hand detailer plan written at Matt's request 2026-10-27; implementation unscheduled. First stage qualifies detector code/weight terms, packaging and detection usefulness. [Plan](docs/plan-detailer.md#stages-and-completion-criteria). |

## Generation, jobs and models

**Owner:** [docs/design-notes.md](docs/design-notes.md) for the deferred items; the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers named in each row.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#6` | DEFERRED | LOW | M+card | Live-preview maps for the four-channel SDXL VAE and FLUX.2's VAE (ERNIE; klein also lacks an unpack hook); previews switch themselves off there. [Behaviour](docs/verified.md#behaviour). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#15` | CARD | LOW | S+card | Peak memory per image: phase 3's FLUX.1 Krea [dev] render, the recorded peak beside `nvidia-smi` and shared memory. [Plan](docs/plan-peak-vram.md). |
| `GEN#16` | CARD | LOW | S+card | Generate then refine: phase 3's FLUX.1 Krea dev night at 1.5x, peak of the alternation and the pair's time. [Plan](docs/plan-then-refine.md). |
| `GEN#20` | DEFERRED | LOW | L+card | Opt-in A1111 weighting and 77-token chunking on SDXL, deferred by Matt 2026-10-27 because he does not write weighted prompts; revisit on request. [10-24 G10](docs/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |

## Speed

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#speed), including each row's CPU-only next planning slice. The measured weight-only fp8/int8 cases run slower a step than bf16; none of the proposed speed tools is used yet. Each pixel-changing choice is recorded for reproduction.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#1` | CARD | MED | L+card | Qualify toolchain/cache and the card; source, bounded diagnostics presentation and Windows browser smokes are done. Public nonempty admission stays closed. [Plan](docs/plan-regional-compilation.md). |
| `PERF#2` | BLOCKED | LOW | M+card | A quantization choice with fp8 activations (torchao `Float8DynamicActivationFloat8WeightConfig`) on the card's fp8 tensor cores; waits on `PERF#1`. [10-25 R2](docs/review-2026-10-25.md#r2-fp8-with-fp8-activations). |
| `PERF#3` | CARD | LOW | M+card | Qualify Native/cuDNN speed, fit and repeatability. Public nonempty admission remains closed. [Plan](docs/plan-attention-backends.md). |
| `PERF#4` | CARD | LOW | L+card | Run the fixed off/on quality, time and memory comparisons, then judge the pairs. Public caching remains closed. [Plan](docs/plan-step-caching.md). |

## Gallery and storage

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#gallery).

| id | status | sev | size | item |
|---|---|---|---|---|
| `GAL#6` | USER | LOW | S | Qualify intended-user Windows credential setup/session and independent-password recovery. OS APIs were stub-tested; no real vault access or gallery activation was performed. [Plan](docs/plan-automatic-backups.md). |

## Prompt analyser

**Owner:** the 2026-09-28 checklist's [What remains](docs/review-2026-09-28-checklist.md#what-remains) for the residues; the design notes' [open questions from the analyser](docs/design-notes.md#open-questions-from-the-analyser) for decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. |

## The page

**Owner:** the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers; the design notes for the rest.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#7` | USER | LOW | M | Hands-on NVDA/Narrator, touch, native notification delivery and actual Windows scaling checks; browser emulation and accessibility-tree inspection do not complete them. [Manual checks](docs/design/validation.md#accessibility-and-desktop-checks). |
| `UI#20` | OPEN | LOW | M | Portable codec and atomic export/import API built; next explicit browser merge workflow and acceptance. [Plan](docs/plan-snippet-portability.md). |

## Development tooling

**Owner:** [regional compilation's CPU dependency preparation](docs/plan-regional-compilation.md#cpu-dependency-preparation).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#8` | OPEN | LOW | S | Prepare optional compiler dependency/pin, lockfile treatment and notices without workstation installation or runtime admission. [Scope and acceptance](docs/plan-regional-compilation.md#cpu-dependency-preparation). |

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
| 2026-10-24 | G1 smoke runner and G2 lint/CI built the same day; G5 choices, G6 snippets, G7 queue, G8 size/crop and G9 help built 2026-10-25/26 ([choices](docs/plan-random-prompt-variables.md), [snippets](docs/plan-prompt-snippets.md), [queue](docs/plan-queue-controls.md), [size/crop](docs/plan-size-axis-crop-anchor.md), [help](docs/plan-shortcuts.md)); G3, G4 and G10 indexed above; the rest recorded as not taken. |
| 2026-10-25 | R1-R10 indexed above. Tooling has [execution stages](docs/plan-tooling.md); speed/backups have CPU-only planning starts. Matt requested both edit plans 2026-10-27; `INP#11` is in progress and detailer implementation remains unscheduled. |
| 2026-10-26 workflow | Three reproduced findings fixed and qualified: mixed prompt prose (`PA#10`), incomplete numbers (`GEN#24`) and unseen queued outcomes (`GEN#25`). [Evidence and results](docs/review-2026-10-26-ux.md). |
| 2026-10-26 keyboard | All three approved stages (`UI#17`-`UI#19`) fixed and qualified, including fast routes and compact setup. [Evidence](docs/plan-shortcut-workflow.md). |
| 2026-10-26 sizing/Bundles | Password association (`UI#14`), clear-count explanation (`UI#15`) and frequent secondary target sizing (`UI#16`) fixed and qualified. [Evidence and results](docs/review-2026-10-26-sizing-bundles.md). |
| 2026-10-26 TODO audit | [Owner/code reconciliation](docs/review-2026-10-26-todo.md): Marian chunk truncation (`GEN#26`), portable screenshots (`DEV#7`), restored critic decision (`PA#11`) and bounded tooling/planning stages. |
| 2026-10-26 overnight | `GEN#26`, `DEV#5`-`DEV#7` built and locally qualified; all four feature plans written. [Tooling evidence](docs/plan-tooling.md). Linux browser qualification (`DEV#4`) closed with PR #15 (hosted, 77 of 77); no GPU acceptance is claimed. |
| 2026-10-27 decisions | Matt deferred `GEN#20`; the requested [instruction-editing](docs/plan-instruction-editing.md) and [detailer](docs/plan-detailer.md) plans are written. Matt then commissioned starting `INP#11` without the card; `INP#12` remains deferred. `PA#11` closed by keeping critic rewrites available with the existing [omission warning](docs/design-notes.md#open-questions-from-the-analyser). |
| 2026-10-28 TODO audit | [Reconciliation and new work](docs/review-2026-10-28-todo.md): corrected stale edit/design status and `CARD#4` scope; added the requested named setups, saved searches, snippet portability and separate compiler preparation. Named setups (`GEN#27`, [plan](docs/plan-generation-presets.md#browser-acceptance-2026-10-28)) and saved gallery searches (`GAL#7`, [plan](docs/plan-saved-searches.md#browser-acceptance)) are built and locally qualified. |

---

## How to read this file

Rules for writing it: read them before editing a row, not before picking one.

**This file is an index, not a work order.** One row per open item: an id, a status, two grades, one clause and a pointer. Why an item matters, how to build it and what counts as done belong in its owner (a plan, the design notes, a review, `docs/verified.md`); a second copy here goes stale unnoticed.

1. **A row leaves this file in the same commit that finishes it.** Record the result in the owner (a measurement in `docs/verified.md` with its date, a build in the plan's status or the design notes) and delete the row; no closed rows stay here. An emptied section becomes one line under "Review coverage" or is dropped.
2. **The owner is the source of truth.** When this file and its owner disagree, this file is stale.
3. **When part of an item lands, replace the clause with what's left;** don't append to it. Keep a row under about 500 bytes.
4. **New work starts in an owner:** a plan in `docs/plan-*.md` for a feature, a review's findings for a defect, the design notes for a deliberate gap. Then it gets a row here.
5. **NOW changes only when its items finish or Matt reorders them.** Refresh its ready count when rows or statuses change.

**`id`:** a stable handle, not an order; a retired id is never reused. The prefix is the section: `CARD` measurements on the card, `INP` inpainting, outpainting and edits, `GEN` generation, jobs and models, `PERF` speed, `GAL` gallery and storage, `PA` prompt analyser, `UI` the page, `DEV` docs and tooling. A review's own id (10-12 F7, U61) is named in the clause.

**`status`**, a closed list:

- `OPEN`: implementation or CPU/browser checks can start; the owner defines completion. `+card` still requires a separately authorized card pass before closure.
- `CARD`: the next step is a run on the RTX 5080. Matt runs it or OKs it; one size at a time, never past the card.
- `USER`: the next step needs Matt's hands-on trial or manual validation; listed in NOW, separately from ready work.
- `DECISION`: the next step is Matt's call. Always listed in NOW.
- `BLOCKED`: waits on another row (named in the clause).
- `DEFERRED`: paused by priority; picked up only when NOW admits it or its condition is met.

**`sev`:** the cost of leaving it open, not difficulty. `HIGH` a built feature with no measurement behind it, or a decision that holds up other work; `MED` a real defect or gap, but bounded; `LOW` convenience, polish, doc debt.

**`size`:** engineer-hours: `S` up to 2 h, `M` 2-6 h, `L` 6-20 h, `XL` over 20 h. `+card` means the real cost is time on the card, which Matt runs or OKs.

**Decided against:** what the docs record as a trade-off or a decision not to build (a review's "Accepted" or "Kept by decision") never gets a row.
