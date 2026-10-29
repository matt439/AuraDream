# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-29):** keep useful work ready for Matt's common workstation tasks. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog (updated 2026-10-30):** 19 `OPEN` rows. `INP#11` stage 4's code is built on `feat/inp11-stage4`; its [card acceptance step](docs/plan-instruction-editing.md#card-acceptance-step) waits on Matt's OK. The [2026-10-29 audit](docs/review-2026-10-29-todo.md)'s rows follow; start with the defects and Often-feature rows: Finish run's missing passes (`GEN#35`), Compare scales (`UI#34`). The card-session tools (`DEV#20`, `DEV#22`, `DEV#23`, `PERF#6`) shorten later card runs; they don't replace them.

**Decisions for Matt:** none open. On 2026-10-29 Matt took the recommended answers for `PERF#7` (cache beside the gallery, cleared on a torch/Triton/driver change, static shapes; shipping waits on `PERF#1`), `PERF#5`, `GAL#21` and `UI#34`; declined `GEN#39` and unattended backups (`GAL#22`, `GAL#6`); kept `GEN#6`, `INP#9` and `INP#10` deferred.

**Needs Matt's hands or card:** workbench task trials (`DEV#1`), hands-on accessibility/desktop checks (`UI#7`), and every `CARD` row, including the speed qualifications `PERF#1`, `PERF#3` and `PERF#4`, and `INP#11`'s acceptance step (one load). `PERF#2`'s card measurement waits on `PERF#1`'s.

**GPU runs are Matt's to run or OK, never past the 16 GB card** (every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence. Tooling rows (`DEV#20`-`DEV#26`): [card harness](docs/plan-card-harness.md) and [tooling](docs/plan-tooling.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#20` | OPEN | MED | M+card | Check in `tools/card_guard.py`: a pure ok/trip/refuse-projected decision (card, shared rise, free RAM/commit, the ERNIE line projection), defaults in one block, JSONL rows, `--watch-pid` tree kill; fixture tests from the 2026-10-25 stops. Stage 2 +card with Matt's OK. [Card guard](docs/plan-card-harness.md#card-guard). |
| `DEV#21` | OPEN | LOW | M | The 2026-10-18 trial kit is stale. The driver types into W/H, which now sit in the closed Custom dimensions disclosure. The fixture (v1) has no saved setup, saved search, snippet or Then refine roles, and no keyboard routes are used. Next: fixture v2, driver repair, cards and a new dated agent baseline; Matt's trial stays `DEV#1`. [Refresh](docs/design/trials/README.md#refresh-for-the-current-app-dev21) |
| `DEV#22` | OPEN | LOW | S | Add `tools/image_compare.py`: content hash, mean/max channel, PSNR, % off >16, differing pixels, outside-mask max, seam step; JSON plus a `--row` in verified.md's column order; tests on synthetic PNGs. [Image metrics](docs/plan-card-harness.md#image-metrics). |
| `DEV#23` | OPEN | LOW | M+card | Check in `tools/card/card10-*.json`: one real-input pack per `CARD#10` fix commit plus the idle Ctrl-C, with no `/api/fake/*` step and a wait for the adapter listing; a structure test and a fake dry run. Stage 2 +card. [CARD#10 step packs](docs/plan-card-harness.md#card10-step-packs). |
| `DEV#24` | OPEN | LOW | S | Add run_smoke.py `--rerun-failed-alone` and `--from-summary`. Exit-1 failures rerun serially, each alone on a fresh fake. summary.json gains `jobs`/`offset`/`rerun`/`verdicts` (passed-alone, failed-alone) and keeps `results` unchanged. The exit stays the first pass's. [Plan](docs/plan-tooling.md#smoke-reruns-alone-2026-10-29-dev24) |
| `DEV#25` | OPEN | LOW | S | Suite and smoke times are invisible. Add `--durations=25` to the CI Test step, and a read-only tools/smoke_timings.py over summary.json, run after Smokes on success or failure. Record Linux CI and Windows local baselines with commit and worker count. [Plan](docs/plan-tooling.md#suite-and-smoke-timings-2026-10-29-dev25) |
| `DEV#26` | OPEN | LOW | S | AGENTS.md's fake-knob list omits /api/fake/model. Add a cold test_docs_maps test that holds all 8 /api/fake routes and 7 fake_server flags to AGENTS.md and docs/rules.md: it fails before the one AGENTS.md line and passes after. [Plan](docs/plan-tooling.md#fake-server-knob-map-2026-10-29-dev26) |
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
| `INP#11` | CARD | LOW | XL+card | Stages 0-3 done; stage 4's ten slices built on local `feat/inp11-stage4` (Kontext listed, the real gate open inside the measured envelope: fp8, model offload, Windows, 1:1 at 512x512). Left: the card acceptance step, one load on the real server and page; the branch stays local until it passes. [Card step](docs/plan-instruction-editing.md#card-acceptance-step). |
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#9` | DEFERRED | LOW | M | Phase 6, the masked area only (`padding_mask_crop` at a working size, a recorded `mask_crop`); not on Z-Image, whose class lacks it. Also for large SDXL outpaint canvases. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |
| `INP#12` | DEFERRED | LOW | L+card | Face/hand detailer plan written at Matt's request 2026-10-27; implementation unscheduled. First stage qualifies detector code/weight terms, packaging and detection usefulness. [Plan](docs/plan-detailer.md#stages-and-completion-criteria). |

## Generation, jobs and models

**Owner:** [docs/design-notes.md](docs/design-notes.md) for the deferred items; the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers named in each row.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#35` | OPEN | MED | M | Then refine: Finish run counts only base cells, so a base saved before a cancelled or failed pass gets no pass, and a run missing only passes says "nothing in this run is missing". Add those passes (pass_sources, one spec per run, preview `passes`). [Plan](docs/plan-then-refine.md#phase-4-finish-run-adds-missing-passes). |
| `GEN#36` | OPEN | LOW | S | Mass refine has no Versions (H4/H5 Often). Add `versions` (1/2/4) as one cell per version at seed+i, the seeds the single dialog's Versions take, so pruning stays per image. versions=1 keeps every key, run id and status. Budget counts images x versions. One strength stays one strength. [Owner](docs/review-2026-10-29-backlog.md#gen36-mass-refine-versions) |
| `GEN#37` | OPEN | LOW | M | Then refine: More seeds of a then-refine run makes bases only. Read the run's retained spec (from the batch's mates' jobs) and queue each new base with its pass under the same pass run id. Add an opt-out tick in the More seeds dialog and a refusal in words for each batch. [Plan](docs/plan-then-refine.md#phase-5-more-seeds-carries-the-pass). |
| `GEN#38` | OPEN | LOW | M | The model picker's Speed shows only retained card samples (K2 Often). Add 'This gallery: ~N s at WxH, S steps (median of n)' to the Speed note. It comes from direct Eager renders, read in the /api/models route (model_info stays cold). Labelled as history, offload noted as unrecorded, never sets the category. [Owner](docs/review-2026-10-29-backlog.md#gen38-gallery-timing-in-the-model-picker) |
| `GEN#15` | CARD | LOW | S+card | Peak memory per image: phase 3's FLUX.1 Krea [dev] render, the recorded peak beside `nvidia-smi` and shared memory. [Plan](docs/plan-peak-vram.md). |
| `GEN#16` | CARD | LOW | S+card | Generate then refine: phase 3's FLUX.1 Krea dev night at 1.5x, peak of the alternation and the pair's time. [Plan](docs/plan-then-refine.md). |
| `GEN#6` | DEFERRED | LOW | M+card | Live-preview maps for the four-channel SDXL VAE and FLUX.2's VAE (ERNIE; klein also lacks an unpack hook); previews switch themselves off there. [Behaviour](docs/verified.md#behaviour). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#20` | DEFERRED | LOW | L+card | Opt-in A1111 weighting and 77-token chunking on SDXL, deferred by Matt 2026-10-27 because he does not write weighted prompts; revisit on request. [10-24 G10](docs/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |

## Speed

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#speed), including each row's CPU-only next planning slice. The measured weight-only fp8/int8 cases run slower a step than bf16; none of the proposed speed tools is used yet. Each pixel-changing choice is recorded for reproduction.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#5` | OPEN | MED | S | Plan the fp8-activation choice split from `PERF#2`, from the installed torchao 0.18 source: sm_120 passes its 8.9 gate, AUTO runs `torch._scaled_mm`, no compile needed, silent layer skips, today's `fp8` is PerRow weight-only; per-component `quant_mapping`, `fp8-dynamic` carry, refusal matrix, three questions for Matt. [Plan](docs/plan-fp8-activations.md). |
| `PERF#6` | OPEN | LOW | S | Fix the shared speed comparison cases before any card run: `tools/perf_cases.py` (Krea dev fp8, 28 steps, CFG 3.5, 1024x1024 and 1360x768, three verbatim author prompts x seeds 4471/8823) plus `tests/test_perf_cases.py` (`applied()` unchanged, recipe-only key difference, `--dry-run` counts, nonempty recipe refused). [Cases](docs/plan-step-caching.md#fixed-comparison-cases). |
| `PERF#1` | CARD | MED | L+card | Qualify toolchain/cache and the card; source, bounded diagnostics presentation and Windows browser smokes are done. Public nonempty admission stays closed. [Plan](docs/plan-regional-compilation.md). |
| `PERF#3` | CARD | LOW | M+card | Qualify Native/cuDNN speed, fit and repeatability. Public nonempty admission remains closed. [Plan](docs/plan-attention-backends.md). |
| `PERF#4` | CARD | LOW | L+card | Run the fixed off/on quality, time and memory comparisons, then judge the pairs. Public caching remains closed. [Plan](docs/plan-step-caching.md). |
| `PERF#2` | BLOCKED | LOW | M+card | The card measurement of a quantization choice with fp8 activations (torchao `Float8DynamicActivationFloat8WeightConfig`); waits on `PERF#1`'s card qualification, and its CPU plan is `PERF#5`. [10-25 R2](docs/review-2026-10-25.md#r2-fp8-with-fp8-activations). |
| `PERF#7` | BLOCKED | LOW | — | Whether the compiled mode ships waits on `PERF#1`'s card table; Matt leans to an opt-in Models choice with Eager the default if it pays off. Cache location, size policy and static shapes are decided. [Questions](docs/plan-regional-compilation.md#matts-questions). |

## Gallery and storage

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#gallery).

| id | status | sev | size | item |
|---|---|---|---|---|
| `GAL#23` | OPEN | LOW | S | A search that finds nothing does not say that a mistyped key such as `sed:42` was searched as prompt words (E5/E9 Often). syncGalleryEmpty names a near SEARCH_KEYS match and offers 'Search seed:42' (no scroll), in the bin too. No change to head height. [Owner](docs/review-2026-10-29-backlog.md#gal23-mistyped-search-keys) |
| `GAL#24` | OPEN | LOW | S | Bin cards do not show their age, although binned_at drives the age-scoped Empty (F5 follow-up). Add 'binned N days ago' (whole days, rounded down) to the card's hover/focus caption and its accessible name, with no new line and no layout change. [F5 follow-up](docs/review-2026-10-12.md#f5-follow-up-bin-card-age-gal24) |

## Prompt analyser

**Owner:** the 2026-09-28 checklist's [What remains](docs/review-2026-09-28-checklist.md#what-remains) for the residues; the design notes' [open questions from the analyser](docs/design-notes.md#open-questions-from-the-analyser) for decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. |

## The page

**Owner:** the 2026-10-12 review's [feature ideas](docs/review-2026-10-12.md#feature-ideas) for the `F` numbers; the design notes for the rest.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#34` | OPEN | MED | M | The Refine dialog cannot compare scales of one keeper. Add Compare scales: size_axis pairs from refine.scaled (1x, 1.5x, 2x), only factors that pass the guard and the tight line, a server preview, a sheet with a Size axis, and the list never remembered as the scale. Quality at 2x is a card stage. [Owner](docs/plan-size-axis-crop-anchor.md#compare-scales-in-the-refine-dialog) |
| `UI#36` | OPEN | LOW | S | No key goes back to the prompt box (L6/A1 Often). Unmodified P, under the same guards as `/` (none in fields, menus, popups or dialogs), focuses `#prompt` with preventScroll and the caret at the end. It changes no text, Undo, Check or picks. Also add a help row in Prompts and `aria-keyshortcuts="P"`. [Stage 4](docs/plan-shortcut-workflow.md#stage-4-prompt-route) |
| `UI#37` | OPEN | LOW | S | Pick this run, which F19 left out of UI#6, is still missing (E12/G5 Often). A card-menu item asks the server for the run's live ids (`run:`/`job:` as Show run as sheet does) and adds them up to selectionCap. It is held with a reason when the row has neither a run nor a job id, and drops a late reply after the bin or Clear. [F19 follow-up](docs/review-2026-10-12.md#f19-follow-up-pick-this-run-ui37) |
| `UI#38` | OPEN | LOW | S | The bundle password fields give no Caps Lock notice (J11/J12 Often). Show 'Caps Lock is on' from getModifierState on the three password inputs. It is aria-live, positioned out of the flow so nothing moves, hidden on blur, and nothing is stored. [Owner](docs/review-2026-10-29-backlog.md#ui38-caps-lock-notice-on-bundle-passwords) |
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
| 2026-10-24 | G1 smoke runner and G2 lint/CI built the same day; G5 choices, G6 snippets, G7 queue, G8 size/crop and G9 help built 2026-10-25/26 ([choices](docs/plan-random-prompt-variables.md), [snippets](docs/plan-prompt-snippets.md), [queue](docs/plan-queue-controls.md), [size/crop](docs/plan-size-axis-crop-anchor.md), [help](docs/plan-shortcuts.md)); G3, G4 and G10 indexed above; the rest recorded as not taken. |
| 2026-10-25 | R1-R10 indexed above. Tooling has [execution stages](docs/plan-tooling.md); speed/backups have CPU-only planning starts. Matt requested both edit plans 2026-10-27; `INP#11` is in progress and detailer implementation remains unscheduled. |
| 2026-10-26 workflow | Three reproduced findings fixed and qualified: mixed prompt prose (`PA#10`), incomplete numbers (`GEN#24`) and unseen queued outcomes (`GEN#25`). [Evidence and results](docs/review-2026-10-26-ux.md). |
| 2026-10-26 keyboard | All three approved stages (`UI#17`-`UI#19`) fixed and qualified, including fast routes and compact setup. [Evidence](docs/plan-shortcut-workflow.md). |
| 2026-10-26 sizing/Bundles | Password association (`UI#14`), clear-count explanation (`UI#15`) and frequent secondary target sizing (`UI#16`) fixed and qualified. [Evidence and results](docs/review-2026-10-26-sizing-bundles.md). |
| 2026-10-26 TODO audit | [Owner/code reconciliation](docs/review-2026-10-26-todo.md): Marian chunk truncation (`GEN#26`), portable screenshots (`DEV#7`), restored critic decision (`PA#11`) and bounded tooling/planning stages. |
| 2026-10-26 overnight | `GEN#26`, `DEV#5`-`DEV#7` built and locally qualified; all four feature plans written. [Tooling evidence](docs/plan-tooling.md). Linux browser qualification (`DEV#4`) closed with PR #15 (hosted, 77 of 77); no GPU acceptance is claimed. |
| 2026-10-27 decisions | Matt deferred `GEN#20`; the requested [instruction-editing](docs/plan-instruction-editing.md) and [detailer](docs/plan-detailer.md) plans are written. Matt then commissioned starting `INP#11` without the card; `INP#12` remains deferred. `PA#11` closed by keeping critic rewrites available with the existing [omission warning](docs/design-notes.md#open-questions-from-the-analyser). |
| 2026-10-28 TODO audit | [Reconciliation and new work](docs/review-2026-10-28-todo.md): corrected stale edit/design status and `CARD#4` scope; added the requested named setups, saved searches, snippet portability and separate compiler preparation. Named setups (`GEN#27`, [plan](docs/plan-generation-presets.md#browser-acceptance-2026-10-28)) and saved gallery searches (`GAL#7`, [plan](docs/plan-saved-searches.md#browser-acceptance)) are built and locally qualified. |
| 2026-10-28 PNG | Named [JSON decoder refusals (`UI#26`)](docs/review-2026-10-28-png-json-refusals.md) and [unreadable-size fallback (`UI#27`)](docs/review-2026-10-28-png-size-refusals.md). |
| 2026-10-28 library | Recovery owners: [snippet/setup libraries and remaining owner audit](docs/review-2026-10-28-library-recovery.md), [setup list recovery (`UI#23`)](docs/review-2026-10-28-setup-list-recovery.md), [saved searches (`UI#24`)](docs/review-2026-10-28-saved-search-recovery.md). |
| 2026-10-28 gallery | Owners: [picked favorites (`GAL#8`)](docs/review-2026-10-28-picked-favorites.md), [favorites (`GAL#9`)](docs/review-2026-10-28-favorite-write-order.md), [exact lifecycle actions (`GAL#10`)](docs/review-2026-10-28-bin-restore-order.md), [Delete unstarred/Empty (`GAL#11`)](docs/review-2026-10-28-broad-bin-order.md), [local date bounds (`GAL#13`)](docs/review-2026-10-28-local-date-bounds.md), [explicit date ranges (`GAL#14`)](docs/plan-gallery-date-ranges.md), [orientation values (`GAL#15`)](docs/plan-gallery-orientation.md), [scientific scalars (`GAL#16`)](docs/review-2026-10-28-numeric-search-scalars.md), [actual area (`GAL#17`)](docs/plan-gallery-area-search.md). |
| 2026-10-28 jobs | Owners: [queued cancellation receipts (`GEN#28`)](docs/review-2026-10-28-queued-cancel-recovery.md), [watched action replies (`GEN#29`)](docs/review-2026-10-28-watched-action-replies.md), [wrapped seed headlines (`GEN#30`)](docs/review-2026-10-28-more-seeds-headline.md). |
| 2026-10-28 tooling | Owners: [fatal fixture prerequisites (`DEV#9`)](docs/review-2026-10-28-fatal-prerequisites.md), [fresh fake directories (`DEV#10`)](docs/review-2026-10-28-fake-fresh.md), [fake credential refusal (`DEV#11`)](docs/review-2026-10-28-fake-credentials.md), [port preflight (`DEV#12`)](docs/review-2026-10-28-smoke-port-plan.md), [fake CLI numbers (`DEV#13`)](docs/review-2026-10-28-fake-cli-numbers.md), [viewport preflight (`DEV#14`)](docs/review-2026-10-28-viewport-preflight.md), [direct port preflight (`DEV#15`)](docs/review-2026-10-28-browser-port-preflight.md), [step containers (`DEV#16`)](docs/review-2026-10-28-browser-step-containers.md), [preparation order (`DEV#17`)](docs/review-2026-10-28-browser-preparation-order.md), [startup geometry (`DEV#18`)](docs/review-2026-10-28-startup-viewport.md), [input refusals (`DEV#19`)](docs/review-2026-10-28-browser-input-refusals.md). |
| 2026-10-29 gallery | [Runs scope and deliberate retry (`GAL#18`)](docs/review-2026-10-28-runs-scope-recovery.md), including actual-handler/native controls and new screenshot inspection. |
| 2026-10-29 grid | [Closed preview ownership (`UI#29`)](docs/review-2026-10-28-grid-preview-close.md), with actual-module/native controls and scoped screenshot inspection. |
| 2026-10-29 copy feedback | [Original image and inspect-session ownership (`UI#30`)](docs/review-2026-10-29-copy-feedback-ownership.md), with synthetic-clipboard native controls and two inspected new screenshots. |
| 2026-10-29 API paging | [SQLite-bound gallery/bin queries (`GAL#20`)](docs/review-2026-10-29-gallery-page-bounds.md), with real overflow controls, legal-MAX binding and existing paging/scope regressions. |
| 2026-10-29 Check dismissal | [Retained findings and dismissed-panel focus (`UI#31`)](docs/review-2026-10-29-check-dismissal.md), with actual-module/original controls, strict fake native checks and three inspected screenshots. |
| 2026-10-29 pending slices | [Bundle & clear ordering (`GAL#12`)](docs/review-2026-10-28-bundle-clear-order.md#final-qualification-2026-10-29), [Reuse-pass opening ownership (`UI#28`)](docs/review-2026-10-28-reuse-pass-opening.md#final-qualification-2026-10-29) and [a split receipt's watched extra (`GEN#31`)](docs/review-2026-10-29-split-extra-queue.md), each with original-source negative controls, an independent review whose confirmed findings are fixed, and inspected new screenshots. Screenshot inspection then closed [Runs favorite counts (`GAL#19`)](docs/review-2026-10-29-runs-favorite-counts.md#screenshot-inspection-2026-10-29) and [snippet portability (`UI#20`)](docs/plan-snippet-portability.md#final-screenshot-inspection-2026-10-29). |
| 2026-10-29 setup portability | [Saved-setup export/import (`UI#25`)](docs/plan-setup-portability.md#stage-3-implementation-and-browser-evidence-2026-10-29): cold codec, atomic API and the browser merge dialog, with an owned-scratch smoke and three inspected new screenshots. |
| 2026-10-29 compiler dependency | `DEV#8` closed: the exact triton-windows wheel's hash, metadata, RECORD and bundled NVIDIA/TinyCC notices were inspected, and a fresh online resolution reproduces the lock. [Results](docs/plan-regional-compilation.md#cpu-dependency-preparation). |
| 2026-10-29 CI browser startup | [Bounded browser startup](docs/review-2026-10-29-ci-browser-startup.md): hosted qualification passed on e687fcd and on main after PR #16. |
| 2026-10-29 TODO audit | [Reconciliation and new work](docs/review-2026-10-29-todo.md): stale owner text fixed; 27 OPEN rows and four new DECISION rows added at Matt's request. The 2026-10-29 closures above are on a local branch, not yet on main or hosted CI. |

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
