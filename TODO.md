# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-18):** keep useful work ready for Matt's common workstation tasks. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog:** five `OPEN` tasks from the [2026-10-24 review](docs/review-2026-10-24.md), each needing a plan first: `GEN#17`-`GEN#19`, `UI#11` and `UI#12`. Deferred features remain deferred.

**Needs Matt:** workbench task trials (`DEV#1`) and hands-on screen-reader/touch checks (`UI#7`) are the next `USER` work. Two decisions wait: whether SDXL prompt weighting is wanted at all (`GEN#20`), and how a cold quantizing FLUX-family load fits this machine's RAM (`GEN#22`), which unblocks the FLUX-family `CARD` rows and `GEN#15`/`GEN#16`. Whole-app layout implementation waits for its trials; it remains a design hypothesis.

**GPU runs are Matt's to run or OK, never past the 16 GB card** (every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

**Proposed 2026-10-25, not yet triaged by Matt:** larger work from the [2026-10-25 review](docs/review-2026-10-25.md). Ready, plan first: speed (`PERF#1`, `PERF#3`, `PERF#4`; `PERF#2` waits on `PERF#1`), tooling (`DEV#4`-`DEV#6`) and automatic backups (`GAL#6`). Two decisions: whether to plan editing by instruction (`INP#11`) and a face and hand detailer (`INP#12`).

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |

## Measurements on the card

**Owner:** [docs/verified.md](docs/verified.md): each row's "Not measured" or "Not verified" list. A result is recorded there, with its date, in the same commit that removes the row here.

| id | status | sev | size | item |
|---|---|---|---|---|
| `CARD#1` | CARD | LOW | S+card | Inpaint at fp8: FLUX.1 dev and schnell at 0.95, a LoRA through a FLUX.1 inpaint, the seam by number on FLUX.1; their loads stop at the RAM guard (`GEN#22`). Nunchaku is `CARD#3`, later families are `INP`. [Inpaint](docs/verified.md#inpaint). |
| `CARD#2` | CARD | LOW | S+card | Outpaint: FLUX.1 dev and schnell, and the outpaint's seam by number on FLUX.1; their loads stop at the RAM guard (`GEN#22`). [Outpaint](docs/verified.md#outpaint). |
| `CARD#3` | CARD | LOW | S+card | An inpaint under WSL through the real server and the Inpaint dialog, and over the FLUX.1 dev and schnell lite transformers. [Under Nunchaku](docs/verified.md#under-nunchaku-wsl-2026-10-25). |
| `CARD#4` | CARD | LOW | M+card | INT8 and fp8 under either offload on FLUX.1 Krea dev, dev, schnell, Krea 2 Turbo and Raw, klein 9B and base 9B (fp8 stopped at the RAM guard on all but Raw and base 9B, int8 on Krea dev; the rest untried, the same load; `GEN#22`); INT8 with sequential offload. Qwen-Image is `CARD#12`. [Quantization and offload](docs/verified.md#quantization-and-offload-2026-10-25). |
| `CARD#6` | CARD | LOW | M+card | Windows fp8 rows: Krea 2 Turbo's time, Krea 2 Raw, FLUX.2 [klein] 9B and base 9B. Krea 2 Turbo's and klein 9B's fp8 loads stop at the RAM guard (`GEN#22`); Raw's and base 9B's were not tried (the same scale). [RAM at the load](docs/verified.md#ram-at-the-load-2026-10-25). |
| `CARD#7` | CARD | LOW | M+card | Refine on FLUX.1 at fp8: dev, what schnell's starting strength keeps, true CFG through FluxImg2Img (their loads stop at the RAM guard, `GEN#22`); offload none and sequential past Z-Image Turbo. [Refine pass](docs/verified.md#refine-pass). |
| `CARD#9` | CARD | LOW | S+card | ERNIE-Image Turbo on fp8: seed reproduction across two processes, whether the 4 MP batch of two fits (the harness's projection refused it), and the page's Expand line and clock over the rewriter's real first-use download. [Behaviour](docs/verified.md#behaviour). |
| `CARD#10` | CARD | LOW | M+card | On the real server, fixes checked only on the fake: 9377de1, 7733f81, 6a7570d, ad01fdc, 2a4d2cc, 2b89ed5 (the FLUX.1 Krea dev load stops at the RAM guard, `GEN#22`); R50's fix (1744dc3) with a typed Ctrl-C, and Ctrl-C to an idle server with a model loaded. [Behaviour](docs/verified.md#behaviour); [10-12 not reached](docs/review-2026-10-12.md#review--2026-10-12). |
| `CARD#11` | CARD | LOW | M+card | The critic: a reorder that reaches the partial-placement check (none of 16 did; every 4B reorder changed words); FLUX.1 Krea [dev] at int8 or sequential beside it, and its rewrites against originals over more seeds (RAM guard, `GEN#22`); kill-on-close under WSL; Ctrl-C during a critique. [Prompt analyser](docs/verified.md#prompt-analyser). |
| `CARD#12` | CARD | LOW | S+card | Official Qwen-Image on Windows: fp8 with sequential offload stops at the RAM guard during the load (`GEN#22`); BF16 with sequential offload is untried. [RAM at the load](docs/verified.md#ram-at-the-load-2026-10-25); [Switching](docs/design-notes.md#switching-checkpoints-while-the-server-stays-open). |

## Inpainting, outpainting and edits

**Owner:** [docs/plan-inpainting.md](docs/plan-inpainting.md#phases) and [docs/plan-outpainting.md](docs/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting). `INP#11` and `INP#12`: the [2026-10-25 review](docs/review-2026-10-25.md#edits---matts-call).

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#9` | DEFERRED | LOW | M | Phase 6, the masked area only (`padding_mask_crop` at a working size, a recorded `mask_crop`); not on Z-Image, whose class lacks it. Also for large SDXL outpaint canvases. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |
| `INP#11` | DECISION | LOW | XL+card | Whether to plan editing an image by instruction (FLUX.1 Kontext [dev]; Qwen-Image-Edit under Nunchaku): a new pass kind from a gallery source, recorded in the chunk and key. [10-25 R9](docs/review-2026-10-25.md#r9-edit-an-image-by-instruction). |
| `INP#12` | DECISION | LOW | L+card | Whether to plan a face and hand detailer (detect, mask per box, the existing inpaint); first, a detector whose licence fits Apache-2.0 (Ultralytics is AGPL-3.0). [10-25 R10](docs/review-2026-10-25.md#r10-a-face-and-hand-detailer). |

## Generation, jobs and models

**Owner:** [docs/design-notes.md](docs/design-notes.md) for the deferred items; the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers named in each row.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#6` | DEFERRED | LOW | M+card | Live-preview maps for the four-channel SDXL VAE and FLUX.2's VAE (ERNIE; klein also lacks an unpack hook); previews switch themselves off there. [Behaviour](docs/verified.md#behaviour). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#15` | CARD | LOW | S+card | Peak memory per image: phase 3's FLUX.1 Krea [dev] render, the recorded peak beside `nvidia-smi` and shared memory; its load stops at the RAM guard (`GEN#22`). [Plan](docs/plan-peak-vram.md). |
| `GEN#16` | CARD | LOW | S+card | Generate then refine: phase 3's FLUX.1 Krea dev night at 1.5x, peak of the alternation and the pair's time; its load stops at the RAM guard (`GEN#22`). [Plan](docs/plan-then-refine.md). |
| `GEN#17` | OPEN | LOW | M | Random-choice prompt variables drawn from each image's seed; plan first. [10-24 G5](docs/review-2026-10-24.md#g5-random-choice-prompt-variables). |
| `GEN#18` | OPEN | LOW | M | Queue: move a job to the front (an admission order of the manager's own, not the gate lock's) and skip the running cell; plan first. [10-24 G7](docs/review-2026-10-24.md#g7-queue-move-to-front-and-skip-one-cell). |
| `GEN#19` | OPEN | LOW | L | Size as a grid axis and a crop anchor for Refine, widening the key once; a run then has more than one size. Plan first. [10-24 G8](docs/review-2026-10-24.md#g8-size-as-a-grid-axis-and-a-crop-anchor-for-refine). |
| `GEN#20` | DECISION | LOW | L+card | Whether to plan opt-in A1111 weighting and 77-token chunking on SDXL; only if Matt writes weighted prompts. [10-24 G10](docs/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |
| `GEN#22` | DECISION | MED | M+card | How a cold quantizing FLUX.1, Krea 2, klein 9B or Qwen-Image load fits this machine's RAM (about 46 GiB at the load, 2026-10-25): free RAM before a run, lower the guard's floor, or quantize a shard at a time (a loader change, plan first). Unblocks the FLUX-family card rows. [The quantizing load's RAM peak](docs/design-notes.md#the-quantizing-loads-ram-peak). |

## Speed

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#speed). Weight-only fp8/int8 run slower a step than bf16 today; none of these is used yet. Each pixel-changing choice is recorded for reproduction.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#1` | OPEN | MED | L+card | Compile the denoiser's repeated blocks under model offload (`compile_repeated_blocks`); Triton on Windows, recompiles per size, LoRA hot-swap, eager-vs-compiled pixels. Plan first. [10-25 R1](docs/review-2026-10-25.md#r1-compile-the-denoisers-repeated-blocks). |
| `PERF#2` | BLOCKED | LOW | M+card | A quantization choice with fp8 activations (torchao `Float8DynamicActivationFloat8WeightConfig`) on the card's fp8 tensor cores; waits on `PERF#1`. [10-25 R2](docs/review-2026-10-25.md#r2-fp8-with-fp8-activations). |
| `PERF#3` | OPEN | LOW | M+card | An attention backend per family (`set_attention_backend`: cuDNN, flash, sage), its Windows wheels and its pixels. Plan first. [10-25 R3](docs/review-2026-10-25.md#r3-attention-backend). |
| `PERF#4` | OPEN | LOW | L+card | Opt-in step caching (First Block Cache, TaylorSeer or MagCache) for the 30-50 step undistilled checkpoints, widening the key once. Plan first. [10-25 R4](docs/review-2026-10-25.md#r4-step-caching-for-the-long-undistilled-schedules). |

## Gallery and storage

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#gallery).

| id | status | sev | size | item |
|---|---|---|---|---|
| `GAL#6` | OPEN | LOW | L | Automatic whole-gallery backup at start-up past a set age, keeping the last N, through the verified export; decide how a password works unattended. Plan first. [10-25 R8](docs/review-2026-10-25.md#r8-automatic-whole-gallery-backup). |

## Prompt analyser

**Owner:** the 2026-09-28 checklist's [What remains](docs/review-2026-09-28-checklist.md#what-remains) for the residues; the design notes' [prompt analyser](docs/design-notes.md#the-prompt-analyser) ("Open questions the build turned up") for the decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. |

## The page

**Owner:** the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers; the design notes for the rest.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#7` | USER | LOW | M | Hands-on screen-reader (NVDA/Narrator) and touch validation; browser emulation and accessibility-tree inspection do not complete it. [Manual checks](docs/design/validation.md#accessibility-and-desktop-checks). |
| `UI#11` | OPEN | LOW | M | Matt's own prompt snippets, inserted through `writeBox`, kept beside the saved settings, with an entry point of no height; plan first. [10-24 G6](docs/review-2026-10-24.md#g6-own-prompt-snippets). |
| `UI#12` | OPEN | LOW | S | A `?` overlay listing the page's keyboard shortcuts. [10-24 G9](docs/review-2026-10-24.md#g9-keyboard-shortcut-overlay). |

## Tooling

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#tooling).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#4` | OPEN | LOW | L | Run the browser smokes in CI on Linux Chrome as a second job; fix font-sensitive assertions properly, keep a failed fake's log. Plan first. [10-25 R5](docs/review-2026-10-25.md#r5-run-the-smokes-in-ci). |
| `DEV#5` | OPEN | LOW | S | A pytest that loads `web/`'s classic scripts in `index.html` order into one Node `vm` context and fails on a redeclared global or a syntax error. [10-25 R6](docs/review-2026-10-25.md#r6-check-webs-shared-global-scope-without-a-browser). |
| `DEV#6` | OPEN | LOW | L | Pyright (basic) over `backend/` and `tools/` in CI; fix or individually suppress the first run's findings. [10-25 R7](docs/review-2026-10-25.md#r7-a-type-checker-over-backend). |

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
| 2026-10-24 | Gaps and tooling: G1 (smoke runner) and G2 (lint, CI) built the same day; G3-G10 indexed above; the rest recorded as not taken. |
| 2026-10-25 | Speed, tooling, backups and edits: R1-R10 indexed above (`PERF`, `DEV#4`-`DEV#6`, `GAL#6`, `INP#11`/`INP#12`), untriaged. |

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
