# TODO: open work

An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-18):** keep useful work ready for Matt's common workstation tasks. The queue below can proceed on CPU and disposable fake-server galleries, independently of whole-app layout trials. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog:** 8 `OPEN` tasks can be completed with CPU/browser evidence. One further `OPEN` item, the upscaler (`GEN#5`), can be built but needs a separate card pass before closure.

1. **Continue a run (`GEN#11`):** build phase 2 in dependency order: persistence, Finish service/API, then page actions and restart smoke. These are stages of one feature. [Execution slices](docs/plan-continue-run.md#phase-2-execution-slices).

**Other ready work:** model cache/download facts (`GEN#1`), Stop Expand (`GEN#2`), design-trial preparation (`DEV#3`) and browser accessibility checks (`UI#9`). Their owners define scope and completion. Optional conveniences remain below; there is no need to promote deferred features to fill this queue.

**Needs Matt:** workbench task trials (`DEV#1`) and hands-on screen-reader/touch checks (`UI#7`) are `USER`, not implementation prerequisites for the queue above. No feature decision is waiting. Whole-app layout implementation waits for its trials; it remains a design hypothesis.

**GPU runs are Matt's to run or OK, never past the 16 GB card** (every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |
| `DEV#3` | OPEN | LOW | M | Prepare repeatable synthetic fixtures, current-app expert baseline and a blank observation record for the task trials. [Preparation](docs/design/validation.md#trial-preparation). |

## Measurements on the card

**Owner:** [docs/verified.md](docs/verified.md): each row's "Not measured" or "Not verified" list. A result is recorded there, with its date, in the same commit that removes the row here.

| id | status | sev | size | item |
|---|---|---|---|---|
| `CARD#1` | CARD | LOW | M+card | Inpaint's remaining native SDXL/Z-Image profiles, SDXL LoRA, seam metrics, blended preview and start-noise display; use the owner's full Not measured list. Nunchaku is `CARD#3`, later families are `INP`. [Inpaint](docs/verified.md#inpaint). |
| `CARD#2` | CARD | LOW | M+card | Outpaint's remaining SDXL/Z-Image profiles, seam metrics and intermediate band widths from the owner's Not measured list. [Outpaint](docs/verified.md#outpaint). |
| `CARD#3` | CARD | MED | S+card | An inpaint over a Nunchaku Z-Image transformer under WSL: the profile allows it, but the class has never run over it. [Inpainting](docs/design-notes.md#inpainting). |
| `CARD#4` | CARD | LOW | L+card | INT8, full BF16 and sequential offload per checkpoint; the picker says INT8 is "not yet compared on this card". [Checkpoints](docs/verified.md#checkpoints). |
| `CARD#5` | CARD | LOW | M+card | Size and batch gaps: Anima Turbo/Aesthetic at 1.5-2 MP and non-square, Kohaku, NetaYume and NewBie above about 1 MP, Illustrious at 1536/2048, Animagine Batch 12 (the cap was raised from a Batch 8 reading). [Checkpoints](docs/verified.md#checkpoints); [MP table](docs/anime-model-guide.md#compare-by-total-megapixels). |
| `CARD#6` | CARD | LOW | M+card | Catalogue entries never loaded on the card: Krea 2 Raw (manual approval), FLUX.2 [klein] 9B, klein base 4B and 9B; Krea 2 Turbo's Windows fp8 row has a peak but no time. [Checkpoints](docs/verified.md#checkpoints). |
| `CARD#7` | CARD | LOW | M+card | Refine gaps: FLUX.1 dev at fp8; what the starting strengths keep on schnell, Z-Image Turbo, Z-Image and Qwen-Image; offload none/sequential; true CFG through FluxImg2Img; review the 2026-10-05 ladder pictures. [Refine pass](docs/verified.md#refine-pass). |
| `CARD#8` | CARD | LOW | S+card | User LoRAs on an SDXL checkpoint: the adapter changes the image, weight 0 is byte-identical to none, unload restores the base hash. So far FLUX.1 Krea dev only. [Behaviour](docs/verified.md#behaviour). |
| `CARD#9` | CARD | LOW | S+card | ERNIE-Image Turbo on the Windows fp8 path, and its Expand (rewriter, Marian translation, the 7 GB first-use download). Measured only under Nunchaku NVFP4. [Behaviour](docs/verified.md#behaviour). |
| `CARD#10` | CARD | LOW | M+card | On the real server, fixes checked only on the fake: 9377de1, 7733f81, 6a7570d, ad01fdc, 2a4d2cc, 2b89ed5, 81781bf; and R50's fix (1744dc3) with a real console Ctrl-C, plus the not_found/offline load failures. [Behaviour](docs/verified.md#behaviour); [10-12 not reached](docs/review-2026-10-12.md#review--2026-10-12). |
| `CARD#11` | CARD | LOW | M+card | The critic for real: a reorder where Check placed only some tags, kill-on-close stopping a real `llama-server`, int8 or sequential offload beside it, and whether rewrites beat originals over more than two seeds. [Prompt analyser](docs/verified.md#prompt-analyser). |
| `CARD#12` | CARD | LOW | S+card | Whether official Qwen-Image (over 16 GB) runs on Windows under sequential offload. [Switching](docs/design-notes.md#switching-checkpoints-while-the-server-stays-open). |

## Inpainting and outpainting

**Owner:** [docs/plan-inpainting.md](docs/plan-inpainting.md#phases) and [docs/plan-outpainting.md](docs/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting).

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#3` | CARD | MED | M+card | FLUX.1 inpaint (Krea dev, dev, schnell): measure memory, strength and preview at 1 MP before enabling the family; its recipe is built. [Phase 3](docs/plan-inpainting.md#phases). |
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#8` | BLOCKED | LOW | L | FLUX.1 Fill [dev] as its own inpaint-only catalogue checkpoint: blocked on `INP#3`, decided once FLUX.1 inpaints on Krea dev have been seen on the card, built if they disappoint. [Question 7](docs/plan-inpainting.md#open-questions-for-the-user). |
| `INP#9` | DEFERRED | LOW | M | Phase 6, the masked area only (`padding_mask_crop` at a working size, a recorded `mask_crop`); not on Z-Image, whose class lacks it. Also for large SDXL outpaint canvases. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |

## Generation, jobs and models

**Owner:** [docs/design-notes.md](docs/design-notes.md) for the deferred items; the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers named in each row.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#1` | OPEN | LOW | M | Models picker: each checkpoint's download size and whether it is already in the Hugging Face cache (10-12 F7). |
| `GEN#2` | OPEN | LOW | M | Stop Expand: abandon its result and stop between download/rewrite/translation stages, retaining the GPU gate while actual GPU work continues (10-12 F8). |
| `GEN#5` | OPEN | LOW | L+card | An opt-in pinned GAN upscaler (R-ESRGAN Anime6B) before the Refine pass, where the WAI and RouWei guides cite one, recorded in the reproduction set (10-12 F14). |
| `GEN#6` | DEFERRED | LOW | M+card | Live-preview maps for the four-channel SDXL VAE and FLUX.2's VAE (ERNIE; klein also lacks an unpack hook); previews switch themselves off there. [Behaviour](docs/verified.md#behaviour). |
| `GEN#7` | DEFERRED | LOW | M | A Hugging Face access check before a switch releases the working model; the catalogue's gate word is static. [Switching](docs/design-notes.md#switching-checkpoints-while-the-server-stays-open). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#11` | OPEN | LOW | L | Continue a run, phase 2: Finish service/API, page actions and reload/restart evidence; persistence is built. [Plan](docs/plan-continue-run.md#phase-2-execution-slices). |

## Gallery and storage

**Owner:** the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) and its [not reached](docs/review-2026-10-12.md#review--2026-10-12) list.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GAL#5` | DEFERRED | LOW | M | Optional age-scoped Empty bin using `binned_at`; independent of Delete picked permanently, not needed for its completion (10-12 F5). |

## Prompt analyser

**Owner:** the 2026-09-28 checklist's [What remains](docs/review-2026-09-28-checklist.md#what-remains) for the residues; the design notes' [prompt analyser](docs/design-notes.md#the-prompt-analyser) ("Open questions the build turned up") for the decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#8` | CARD | LOW | S+card | Whether NewBie reads `<name>` as it reads the authors' `<n>`; the XML rule accepts both. |
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. |

## The page

**Owner:** the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers; the design notes for the rest.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#1` | OPEN | LOW | M | "Compare picked": two picked images in the lightbox's swipe and side by side, labelled by the fields that differ (10-12 F9). |
| `UI#2` | OPEN | LOW | S | An opt-in desktop notification when a job finishes in a hidden tab (10-12 F10); image download is `UI#10`, Copy image already exists in the context menu. |
| `UI#3` | OPEN | LOW | S | "Seeds to grid": fill the grid's Seed list from the picked images' seeds (10-12 F11). |
| `UI#7` | USER | LOW | M | Hands-on screen-reader (NVDA/Narrator) and touch validation; browser emulation and accessibility-tree inspection do not complete it. [Manual checks](docs/design/validation.md#accessibility-and-desktop-checks). |
| `UI#8` | DEFERRED | LOW | M | At 1536x864 (1080p at 125%) a running batch's progress falls below the left pane; fitting it means tightening every field under 1800 px. [Panel rows](docs/design-notes.md#the-panels-sampling-and-output-rows). |
| `UI#9` | OPEN | LOW | M | Exercise current-app keyboard/focus, forced colours and reduced motion with synthetic fixtures and real browser events; retain gaps separately from manual checks. [Browser checks](docs/design/validation.md#accessibility-and-desktop-checks). |

## Review coverage

Each review's remaining work is indexed above; closed findings stay in their owners.

| review | disposition |
|---|---|
| 2026-08-29, 2026-09-07 | Every lead re-checked in 2026-09-07's findings; its checklist is all ticked. |
| 2026-09-28 | Its checklist; its residues were built 2026-10-17 (`PA#1`-`PA#4`). F47, the CLI's literal `{word}` and the other entries under "Accepted, and worded as trade-offs" are decided, not open. |
| 2026-10-03 | All confirmed findings and P1 fixed 2026-10-04. |
| 2026-10-04 UI | All done but U61 B (`GEN#8`); U5, U12, U79, U100 and U108 were left by decision. |
| 2026-10-05 areas | 84 fixed, A68 partly by decision; the integration review's K1-K5 fixed. |
| 2026-10-12 | R1-R53 fixed; F1-F4, F16 and F17 built. F6's supported inpaint/outpaint paths are built; remaining families/measurements are `INP`/`CARD`. F5, F7-F15 and F18-F19 are indexed above. |

---

## How to read this file

Rules for writing it: read them before editing a row, not before picking one.

**This file is an index, not a work order.** One row per open item: an id, a status, two grades, one clause and a pointer. Why an item matters, how to build it and what counts as done belong in its owner (a plan, the design notes, a review, `docs/verified.md`); a second copy here goes stale unnoticed.

1. **A row leaves this file in the same commit that finishes it.** Record the result in the owner (a measurement in `docs/verified.md` with its date, a build in the plan's status or the design notes) and delete the row; no closed rows stay here. An emptied section becomes one line under "Review coverage" or is dropped.
2. **The owner is the source of truth.** When this file and its owner disagree, this file is stale.
3. **When part of an item lands, replace the clause with what's left;** don't append to it. Keep a row under about 500 bytes.
4. **New work starts in an owner:** a plan in `docs/plan-*.md` for a feature, a review's findings for a defect, the design notes for a deliberate gap. Then it gets a row here.
5. **NOW changes only when its items finish or Matt reorders them.** Refresh its ready count when rows or statuses change.

**`id`:** a stable handle, not an order; a retired id is never reused. The prefix is the section: `CARD` measurements on the card, `INP` inpainting and outpainting, `GEN` generation, jobs and models, `GAL` gallery and storage, `PA` prompt analyser, `UI` the page, `DEV` docs and tooling. A review's own id (10-12 F7, U61) is named in the clause.

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
