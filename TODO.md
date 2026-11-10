# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-29):** keep useful work ready for Matt's common workstation tasks. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog (updated 2026-11-10):** 16 `OPEN` rows, all from the [2026-11-10 peer review](docs/review-2026-11-10-peers.md): `GEN#50`, `GEN#51`, `GEN#53`-`GEN#57`, `DEV#32`, `DEV#33`, `DEV#37`-`DEV#40` and `UI#41`-`UI#43`. Next: `DEV#32`, the LAN passcode Matt approved (its design note first). Built 2026-11-10 on `todo/peers-batch`, each with a Built note under its finding: `DEV#31` (anti-framing headers), `DEV#34` (the before-update copy), `DEV#35` (the log file), `DEV#36` (the running commit) and `GEN#52` (a full disk pauses the job). Before that review there were none: `GEN#45`, newer checkpoints, was built 2026-11-09 and merged (PR #61): the diffusers 0.42.1 bump, FLUX 3 [Klein] and Klein Base, and FLUX 3's prompt upsampler as an opt-in Expand. The bump was measured on the card 2026-11-10: four shipped checkpoints and Kontext's envelope pixel-identical to 0.40.0, so Kontext edits open again on 0.42.1 ([results](docs/verified.md#diffusers-0421-against-0400-2026-11-10)). FLUX 3 was measured on the card the same day ([results](docs/verified.md#flux-3-klein-and-its-prompt-upsampler-2026-11-10)): Klein fits with room at fp8 and int8 and Klein Base at fp8 (6.08 GiB allocated plain, 7.40 with a reference), Klein at bf16 only without a reference; the upsampler's Expand works, but its first use needs about 41 GiB of RAM with Klein loaded. Left: `GEN#46`'s CI account and code follow-ups (`USER`), and the 3070's int4 (`CARD#21`, deferred) ([bump](docs/design-notes.md#newer-checkpoints-and-the-diffusers-042-bump), [FLUX 3](docs/design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders)). `UI#40`'s remaining code and browser checks are built; phone/platform acceptance is `USER` ([phone review](docs/review-2026-11-05-phone.md#10-edit-dialog-keyboard-acceptance-2026-11-07)). The native Linux server's 8 GB RTX 3070 rows were built 2026-11-05 (its card steps are `CARD#14`-`CARD#16`): `GEN#42`, the Nunchaku format by the card, `GEN#43`, the fit badge by the card's memory, and `GEN#44`, the measured refine and inpaint words named as the 16 GB card's ([8 GB card](docs/design-notes.md#native-linux-a-server-on-the-network)). `INP#9`, only the masked area, was built 2026-11-03 at Matt's request ([plan phase 6](docs/plan-inpainting.md#phases)); its card measurements are `CARD#13`. The [first-run review](docs/review-2026-11-01-first-run.md#stages)'s four stages were built 2026-11-01: docs (`DEV#27`), first-visit panel (`UI#39`), plain-words failures (`GEN#40`) and the installer (`DEV#28`: `install.cmd`, `update.cmd`, `backend/doctor.py`; tested as dry runs and with stub tools, never on a clean machine); its stage 5 (`DEV#29`) stays deferred. `DEV#25` closed with the Linux and Windows CI baselines from pull_request run 41159120243 ([results](docs/plan-tooling.md#results)). `INP#11` stage 4's code is on main (PR #17); its [card acceptance step](docs/plan-instruction-editing.md#card-acceptance-step) waits on Matt's OK. Built 2026-10-30 (PR #22): `GEN#36`-`GEN#38`, `GAL#23`, `DEV#26`, the trial kit refresh (`DEV#21`), and the CPU stages of the card guard (`DEV#20`) and step packs (`DEV#23`), whose card stages are now `CARD` rows.

**Decisions for Matt:** On 2026-11-10 Matt accepted four points of the first built rows: `DEV#31`'s pinned `.js`/`.css` types, `DEV#34`'s same-commit copy left unrefreshed, `DEV#36`'s `linux/control.sh status` line, and the CLI's full-disk hold as a row (`GEN#57`). `GEN#48`, which non-refining checkpoint (Krea 2, Anima, FLUX.2 klein, FLUX 3 Klein) to build an app-owned Refine for first; asked 2026-11-10, Matt not yet sure. The same day Matt said yes to an opt-in passcode for the LAN service (`DEV#32`) and asked for every kept finding of the peer review as a row. On 2026-11-09 Matt started `GEN#45`: bump to diffusers 0.42.1 now; add FLUX 3 Image Klein (distilled, 16 steps) and Klein Base (50 steps, CFG), not Qwen-Image-2.1, 2.1-Turbo or 2512; FLUX 3's 9.4B prompt upsampler only as an opt-in Expand-style step, never automatic; Kontext edits held on the measured 0.40.0 until a card re-measure (done 2026-11-10: the envelope is now 0.42.1). On 2026-11-03 Matt closed `DEV#30`, rating saved searches, Edit by instruction, Only the masked area and Then refine Sometimes and taking the recommended answers to the [persona review](docs/review-2026-11-03-personas.md#open-for-matt)'s other questions (no first-run persona, the watch choice in UX#7 Proposed, Performance collapsed in Models, Finish run's counts on the banner, the backup section hidden; Cancel's width kept), built on `design/dev30-decisions`. The same day Matt said yes to `GEN#41` (built: a fresh visit without a Hugging Face token preselects Z-Image Turbo) and kept `DEV#20`'s commit-headroom floor at 4 GiB, so a card session runs on a quiet machine. On 2026-10-29 Matt took the recommended answers for `PERF#7` (cache beside the gallery, cleared on a torch/Triton/driver change, static shapes; shipping waits on `PERF#1`), `PERF#5`, `GAL#21` and `UI#34`; declined `GEN#39` and unattended backups (`GAL#22`, `GAL#6`); kept `GEN#6`, `INP#9` and `INP#10` deferred. On 2026-11-03 Matt reopened `INP#9` (only the masked area, as the app's crop on FLUX.1, SDXL and Z-Image, inpaint and outpaint): built that day on `inp9/masked-area-only`; its card step is `CARD#13`.

**Needs Matt's hands or card:** `GEN#46`, accepting the FLUX 3 [Klein] licences on CI's Hugging Face account (his own accepted 2026-11-10); `UI#40`'s phone/platform acceptance, workbench task trials (`DEV#1`, kit refreshed 2026-10-30), the typed Ctrl-C on the fake for `DEV#23`, hands-on accessibility/desktop checks (`UI#7`), and every `CARD` row, including the speed qualifications `PERF#1`, `PERF#3` and `PERF#4`, and `INP#11`'s acceptance step (one load). `PERF#2`'s card measurement waits on `PERF#1`'s.

**GPU runs are Matt's to run or OK, never past the card's memory** (16 GB on the 5080, 8 GB on the 3070; every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence. Tooling rows (`DEV#20`, `DEV#23`): the [card harness](docs/plan-card-harness.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#20` | CARD | MED | S+card | `tools/card_guard.py` stage 1 built 2026-10-30. Left: stage 2, one light SDXL fp8 session with the guard beside its own watcher, with Matt's OK, on a quiet machine (floor kept at 4 GiB, 2026-11-03). [Card guard](docs/plan-card-harness.md#card-guard). |
| `DEV#23` | CARD | LOW | S+card | The eight `tools/card/card10-*.json` packs and their test are built 2026-10-30; the six page packs pass on the fake. Left: Matt types Ctrl-C on the fake for the two Ctrl-C packs, then stage 2 on the card. [CARD#10 step packs](docs/plan-card-harness.md#card10-step-packs). |
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |

## First-run experience

**Owner:** [docs/review-2026-11-01-first-run.md](docs/review-2026-11-01-first-run.md): findings F1-F8, the stages in order, and what was not taken.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#29` | DEFERRED | LOW | S | Stage 5: hand-cut version tags for `update.cmd` to follow, and a Pinokio script; after `DEV#28`. [Stage 5](docs/review-2026-11-01-first-run.md#stages). |

## Server, security and tooling

**Owner:** [docs/review-2026-11-10-peers.md](docs/review-2026-11-10-peers.md): findings S1-S7 and T1-T3, each with its first step and what was cut.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#32` | OPEN | MED | L | Opt-in passcode when bound beyond loopback (Matt's yes, 2026-11-10): API, SSE, mounts, `sw.js`, `--via-server`, the fake, a sign-in page. A design note listing every surface comes first. [S2](docs/review-2026-11-10-peers.md#s2-passcode-beyond-loopback). |
| `DEV#33` | OPEN | MED | M | An HTTPS route for LAN and phone: a public-port or allowed-origins setting for the origin check, optional certificate and key, a documented recipe. [S3](docs/review-2026-11-10-peers.md#s3-an-https-route). |
| `DEV#37` | OPEN | MED | M | A stable `code` on the refusals the page acts on (unknown image, binned, More seeds' cap, missing LoRA); the page matches English wording in over twenty places. [S7](docs/review-2026-11-10-peers.md#s7-error-codes-the-page-reads). |
| `DEV#38` | OPEN | MED | M | `browser.mjs` records then fails on `Runtime.exceptionThrown`; a page `error`/`unhandledrejection` line; a no-undef pass over the classic scripts. [T1](docs/review-2026-11-10-peers.md#t1-the-smoke-driver-hears-page-errors). |
| `DEV#39` | OPEN | LOW | M | shellcheck and PSScriptAnalyzer over the scripts (locally first), then a hosted run of the installers up to the GPU check; a CI change needs the ruleset edit. [T2](docs/review-2026-11-10-peers.md#t2-the-scripts-in-ci). |
| `DEV#40` | OPEN | LOW | M | `tools/a11y_audit.js` in `phone_audit.js`'s shape, injected into dialog-heavy smokes; a regression guard, not `UI#7`'s evidence. [T3](docs/review-2026-11-10-peers.md#t3-an-injected-accessibility-audit). |

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
| `CARD#13` | CARD | LOW | M+card | Only the masked area (built 2026-11-03, was `INP#9`): FLUX.1 cropped on a 2.4-8.4 MP source (peak near 12.2/12.8 GiB), seams at padding 0, 32 and 64, Z-Anime cropped at 4.2 MP, an SDXL 7.5 MP outpaint cropped for softness. One load per session. [Not measured](docs/verified.md#only-the-masked-area-built-2026-11-03-not-measured). |
| `CARD#14` | CARD | MED | S+card | On the 3070 server: a suspend with a model loaded, and whether the loaded pipeline survives the resume (`NVreg_PreserveVideoMemoryAllocations=1` is set). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#15` | CARD | LOW | S+card | Nunchaku int4 on the 3070 (sm_86): whether the lite-infer kernels load and render, its time and peak. [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#16` | CARD | LOW | S+card | The prompt critic's pinned CUDA 13.4 llama.cpp build under the 3070's 13.2 driver (first use, a critique). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#21` | DEFERRED | LOW | M+card | FLUX 3 on the 8 GB 3070 as int4/Nunchaku: first wire a load from lite-infer's int4 subfolders (its index points at nvfp4, which sm_86 cannot run), then fit and time; the upsampler cannot fit. FLUX 3 works on the 5080 (2026-11-10): fp8 peaked at 6.08 GiB allocated / 7.32 reserved on a plain image (about 11 GB of the card with its desktop); its fit on the 8 GB 3070 is unmeasured. int4 remains the planned route; build it only if Matt wants FLUX 3 there. [FLUX 3](docs/design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders). |

## Inpainting, outpainting and edits

**Owner:** [docs/plan-inpainting.md](docs/plan-inpainting.md#phases) and [docs/plan-outpainting.md](docs/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting). `INP#11` and `INP#12`: the [instruction-editing plan](docs/plan-instruction-editing.md) and [detailer plan](docs/plan-detailer.md), requested 2026-10-27.

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#11` | CARD | LOW | XL+card | Stages 0-3 done; stage 4's ten slices on main since PR #17 (Kontext listed, the real gate open inside the measured envelope: fp8, model offload, Windows, 1:1 at 512x512). Left: the card acceptance step, one load on the real server and page (the envelope re-measured on diffusers 0.42.1, 2026-11-10). [Card step](docs/plan-instruction-editing.md#card-acceptance-step). |
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |
| `INP#12` | DEFERRED | LOW | L+card | Face/hand detailer plan written at Matt's request 2026-10-27; implementation unscheduled. First stage qualifies detector code/weight terms, packaging and detection usefulness. [Plan](docs/plan-detailer.md#stages-and-completion-criteria). |
| `INP#13` | BLOCKED | LOW | L+card | After `INP#11`'s card acceptance: open the plan's frozen aspects under the 512² area, and a second frozen recipe at 1024x1024 1:1 with its own one-load card protocol. [R3](docs/review-2026-11-10-peers.md#r3-a-wider-instruction-edit-envelope). |

## Generation, jobs and models

**Owner:** the plan or section each row links; `GEN#48`-`GEN#56`: the [2026-11-10 peer review](docs/review-2026-11-10-peers.md)'s R1-R10. Otherwise: [peak memory](docs/plan-peak-vram.md) and [Then refine](docs/plan-then-refine.md) for the card rows; the design notes, [docs/verified.md](docs/verified.md) or a review for the deferred ones.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#15` | CARD | LOW | S+card | Peak memory per image: phase 3's FLUX.1 Krea [dev] render, the recorded peak beside `nvidia-smi` and shared memory. [Plan](docs/plan-peak-vram.md). |
| `GEN#16` | CARD | LOW | S+card | Generate then refine: phase 3's FLUX.1 Krea dev night at 1.5x, peak of the alternation and the pair's time. [Plan](docs/plan-then-refine.md). |
| `GEN#6` | DEFERRED | LOW | M+card | Live-preview maps for the four-channel SDXL VAE and FLUX.2's VAE (ERNIE; klein also lacks an unpack hook); previews switch themselves off there. [Behaviour](docs/verified.md#behaviour). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#20` | DEFERRED | LOW | L+card | Opt-in A1111 weighting and 77-token chunking on SDXL, deferred by Matt 2026-10-27 because he does not write weighted prompts; revisit on request. [10-24 G10](docs/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |
| `GEN#46` | USER | LOW | S | Accept both FLUX 3 [Klein] licences on CI's `HF_TOKEN` account. The local account downloads both repositories since 2026-11-10, and `flux3_loading.preflight` passed on both snapshots (Base's `flow_map: false` included). Then record shard maps in `model-downloads.json`, and move FLUX 3 to `REAL` in `tests/test_prompt_encoding.py` and the CI tokenizer fetch. [FLUX 3](docs/design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders). |
| `GEN#47` | DEFERRED | LOW | S | lite-infer renamed its re-uploads (seen 2026-11-09); the catalogue's ids resolve only through an HTTP 307. Move `launch._LITE` and ERNIE's id to the new names, old ids still read from saved settings, and test `hub_access` and downloads against a redirect. [Renames](docs/design-notes.md#newer-checkpoints-and-the-diffusers-042-bump). |
| `GEN#48` | DECISION | MED | L+card | An app-owned Refine for a family diffusers gives none (Krea 2, Anima, FLUX.2 klein, FLUX 3 Klein), one family first; Matt names which. [R1](docs/review-2026-11-10-peers.md#r1-refine-on-the-families-that-have-none). |
| `GEN#49` | CARD | LOW | L+card | Tiled refine past the one-pass size guards. First a throwaway card look, with Matt's OK: one 2x source as four overlapping masked-area regions at strength 0.3-0.4; a plan only if the seams pass. [R2](docs/review-2026-11-10-peers.md#r2-tiled-refine). |
| `GEN#50` | OPEN | LOW | M | Opt-in unload after N idle minutes (`AURADREAM_IDLE_UNLOAD_MINUTES`, 0 off) on `KeepAwake`'s activity clock; a design note first. [R4](docs/review-2026-11-10-peers.md#r4-release-the-card-after-idle-minutes). |
| `GEN#51` | OPEN | LOW | M | A Windows keep-awake holder behind `platform_inhibitor`, with its status field and doctor row; nothing is held off Linux today. [R5](docs/review-2026-11-10-peers.md#r5-keep-awake-on-windows). |
| `GEN#53` | OPEN | LOW | M | Use prompt in the image's context menu (through `rewriteBox`); Recent prompts under Examples only if that falls short. [R7](docs/review-2026-11-10-peers.md#r7-prompt-recall). |
| `GEN#54` | OPEN | LOW | M | A third snippet target appending a saved `name = a \| b` block to the grid's variables box; a plan addendum first. [R8](docs/review-2026-11-10-peers.md#r8-a-snippet-target-for-grid-variables). |
| `GEN#55` | OPEN | LOW | L | Storage view: a read-only route of measured cache bytes per catalogue entry and what none references; the guarded delete is designed after its output is seen. [R9](docs/review-2026-11-10-peers.md#r9-storage-view). |
| `GEN#56` | OPEN | LOW | M | Inpaint, references, LoRA and the other flags in `details_for`'s notes, tested over the catalogue, and a generated capability table in `docs/usage.md`. [R10](docs/review-2026-11-10-peers.md#r10-capability-flags-per-checkpoint). |
| `GEN#57` | OPEN | LOW | S | The CLI's own save loop (`generate.py`) still fails the cell on a full disk; give it `GEN#52`'s hold through `jobs.disk_full`. [R6](docs/review-2026-11-10-peers.md#r6-a-full-disk-pauses-the-job). |

## Speed

**Owner:** the [2026-10-25 review](docs/review-2026-10-25.md#speed), including each row's CPU-only next planning slice. The measured weight-only fp8/int8 cases run slower a step than bf16; none of the proposed speed tools is used yet. Each pixel-changing choice is recorded for reproduction.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#1` | CARD | MED | L+card | Qualify toolchain/cache and the card; source, bounded diagnostics presentation and Windows browser smokes are done. Public nonempty admission stays closed. [Plan](docs/plan-regional-compilation.md). |
| `PERF#3` | CARD | LOW | M+card | Qualify Native/cuDNN speed, fit and repeatability. Public nonempty admission remains closed. [Plan](docs/plan-attention-backends.md). |
| `PERF#4` | CARD | LOW | L+card | Run the fixed off/on quality, time and memory comparisons, then judge the pairs. Public caching remains closed. [Plan](docs/plan-step-caching.md). |
| `PERF#2` | BLOCKED | LOW | M+card | The card measurement of a quantization choice with fp8 activations (torchao `Float8DynamicActivationFloat8WeightConfig`); waits on `PERF#1`'s card qualification. Its CPU plan is done (`fp8-dynamic`, PerTensor, compared against fp8 only); the CPU source slice and card run are its stages 2-3. [Plan](docs/plan-fp8-activations.md); [10-25 R2](docs/review-2026-10-25.md#r2-fp8-with-fp8-activations). |
| `PERF#7` | BLOCKED | LOW | — | Whether the compiled mode ships waits on `PERF#1`'s card table; Matt leans to an opt-in Models choice with Eager the default if it pays off. Cache location, size policy and static shapes are decided. [Questions](docs/plan-regional-compilation.md#matts-questions). |

## Prompt analyser

**Owner:** the design notes' [prompt analyser](docs/design-notes.md#the-prompt-analyser) for the deferred item; its [open questions](docs/design-notes.md#open-questions-from-the-analyser) for decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. [Prompt analyser](docs/design-notes.md#the-prompt-analyser). |

## The page

**Owner:** [docs/design/validation.md](docs/design/validation.md) for the manual checks; [the phone review](docs/review-2026-11-05-phone.md) for the phone redesign.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#40` | USER | MED | M | Phone acceptance: physical iPhone keyboard/sharing/gestures, Android/system and native iOS edge Back, landscape instruction caret and actual iPad Split View. [Phone review](docs/review-2026-11-05-phone.md#10-edit-dialog-keyboard-acceptance-2026-11-07). |
| `UI#7` | USER | LOW | M | Hands-on NVDA/Narrator, touch, native notification delivery and actual Windows scaling checks; browser emulation and accessibility-tree inspection do not complete them. [Manual checks](docs/design/validation.md#accessibility-and-desktop-checks). |
| `UI#41` | OPEN | LOW | M | The lightbox preloads the next and previous PNG and shows the thumbnail meanwhile; time a 5,000-row gallery on a fake before touching card containment. [T4](docs/review-2026-11-10-peers.md#t4-the-lightbox-loads-its-neighbours). |
| `UI#42` | OPEN | LOW | S | Phone Share offers Picture only (a canvas re-encode) beside With settings; the PNG carries prompt, settings and mask today. [T5](docs/review-2026-11-10-peers.md#t5-phone-share-without-the-settings). |
| `UI#43` | OPEN | LOW | M | Serve `docs/usage.md` read-only and link it from the shortcuts dialog, with a test that linked anchors exist; the renderer is the first choice to make. [T6](docs/review-2026-11-10-peers.md#t6-the-guide-from-the-page). |

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
| 2026-10-28 gallery | `GAL#8`-`GAL#11`, `GAL#13` and `GAL#16` (picked favorites, favorite write order, exact and broad bin order, local date bounds, numeric scalars) are owned by their `docs/review-2026-10-28-*.md`; the search slices by their plans: [date ranges (`GAL#14`)](docs/plan-gallery-date-ranges.md), [orientation (`GAL#15`)](docs/plan-gallery-orientation.md), [area (`GAL#17`)](docs/plan-gallery-area-search.md). |
| 2026-10-28 jobs | Owners: [queued cancellation receipts (`GEN#28`)](docs/review-2026-10-28-queued-cancel-recovery.md), [watched action replies (`GEN#29`)](docs/review-2026-10-28-watched-action-replies.md), [wrapped seed headlines (`GEN#30`)](docs/review-2026-10-28-more-seeds-headline.md). |
| 2026-10-28 tooling | `DEV#9`-`DEV#19` built, each owned by its `docs/review-2026-10-28-*.md`: fatal prerequisites, fresh fake directories, fake credentials, smoke port plan, fake CLI numbers, viewport and browser port preflights, step containers, preparation order, startup viewport and input refusals. [First of them](docs/review-2026-10-28-fatal-prerequisites.md). |
| 2026-10-29 gallery | [Runs scope and deliberate retry (`GAL#18`)](docs/review-2026-10-28-runs-scope-recovery.md), including actual-handler/native controls and new screenshot inspection. |
| 2026-10-29 grid | [Closed preview ownership (`UI#29`)](docs/review-2026-10-28-grid-preview-close.md), with actual-module/native controls and scoped screenshot inspection. |
| 2026-10-29 copy feedback | [Original image and inspect-session ownership (`UI#30`)](docs/review-2026-10-29-copy-feedback-ownership.md), with synthetic-clipboard native controls and two inspected new screenshots. |
| 2026-10-29 API paging | [SQLite-bound gallery/bin queries (`GAL#20`)](docs/review-2026-10-29-gallery-page-bounds.md), with real overflow controls, legal-MAX binding and existing paging/scope regressions. |
| 2026-10-29 Check dismissal | [Retained findings and dismissed-panel focus (`UI#31`)](docs/review-2026-10-29-check-dismissal.md), with actual-module/original controls, strict fake native checks and three inspected screenshots. |
| 2026-10-29 pending slices | Qualified with negative controls, an independent review and inspected screenshots: [Bundle & clear (`GAL#12`)](docs/review-2026-10-28-bundle-clear-order.md), [Reuse-pass opening (`UI#28`)](docs/review-2026-10-28-reuse-pass-opening.md), [split extra (`GEN#31`)](docs/review-2026-10-29-split-extra-queue.md); screenshots then closed [`GAL#19`](docs/review-2026-10-29-runs-favorite-counts.md) and [`UI#20`](docs/plan-snippet-portability.md). |
| 2026-10-29 setup portability | [Saved-setup export/import (`UI#25`)](docs/plan-setup-portability.md#stage-3-implementation-and-browser-evidence-2026-10-29): cold codec, atomic API and the browser merge dialog, with an owned-scratch smoke and three inspected new screenshots. |
| 2026-10-29 compiler dependency | `DEV#8` closed: the exact triton-windows wheel's hash, metadata, RECORD and bundled NVIDIA/TinyCC notices were inspected, and a fresh online resolution reproduces the lock. [Results](docs/plan-regional-compilation.md#cpu-dependency-preparation). |
| 2026-10-29 CI browser startup | [Bounded browser startup](docs/review-2026-10-29-ci-browser-startup.md): hosted qualification passed on e687fcd and on main after PR #16. |
| 2026-10-30 overnight | Built and merged in PR #22: `GEN#36`-`GEN#38`, `GAL#23` (the Gallery section emptied), `DEV#26`, `DEV#21` and stage 1 of `DEV#20`/`DEV#23`, then an integration review and load-flake fixes (polls for fixed sleeps, a bundle settle's hold, profile cleanup, sidecar waits). [Owners](docs/review-2026-10-29-backlog.md); [card harness](docs/plan-card-harness.md). |
| 2026-10-29 TODO audit | [Reconciliation and new work](docs/review-2026-10-29-todo.md): stale owner text fixed; 27 OPEN rows and four new DECISION rows added at Matt's request. |
| 2026-10-31 code hygiene | Stale docs, dead code, duplicates and long files on `chore/code-hygiene`; no row opened. The splits and retained long files are [source-audit.md's third pass](docs/source-audit.md#third-pass-2026-10-31). |
| 2026-11-01 first run | [First-time setup and an installer](docs/review-2026-11-01-first-run.md): `DEV#27`-`DEV#29`, `UI#39`, `GEN#40`, `GEN#41`; the token wizard, release ZIP, user-wide data relocation and signed/bundled apps recorded as not taken. |
| 2026-11-03 personas | [Personas and stories against the app](docs/review-2026-11-03-personas.md): design docs and usage.md reconciled; rerun-preview notes, cross-tab Stopping…, scope titles and held Orientation fixed; its six questions answered 2026-11-03 and the placements built (`DEV#30`, closed). |
| 2026-11-05 phone | [Phone redesign](docs/review-2026-11-05-phone.md#9-remaining-implementation-findings-2026-11-07): all implementation phases and remaining code fixes built; browser and native image-pass keyboard evidence retained. Physical/platform acceptance remains `UI#40` (`USER`). |
| 2026-11-10 peers | [The app against comparable projects](docs/review-2026-11-10-peers.md): 48 candidates checked against the code and earlier decisions, 23 kept (`GEN#48`-`GEN#56`, `INP#13`, `DEV#31`-`DEV#40`, `UI#41`-`UI#43`), the rest recorded as not taken; `DEV#31`, `DEV#34`-`DEV#36` and `GEN#52` built 2026-11-10, the others indexed above. |

---

## How to read this file

Rules for writing it: read them before editing a row, not before picking one.

**This file is an index, not a work order.** One row per open item: an id, a status, two grades, one clause and a pointer. Why an item matters, how to build it and what counts as done belong in its owner (a plan, the design notes, a review, `docs/verified.md`); a second copy here goes stale unnoticed.

1. **A row leaves this file in the same commit that finishes it.** Record the result in the owner (a measurement in `docs/verified.md` with its date, a build in the plan's status or the design notes) and delete the row; no closed rows stay here. An emptied section becomes one line under "Review coverage" or is dropped.
2. **The owner is the source of truth.** When this file and its owner disagree, this file is stale.
3. **When part of an item lands, replace the clause with what's left;** don't append to it. Keep a row under about 500 bytes.
4. **New work starts in an owner:** a plan in `docs/plan-*.md` for a feature, a review's findings for a defect, the design notes for a deliberate gap. Then it gets a row here.
5. **NOW changes only when its items finish or Matt reorders them.** Refresh its ready count when rows or statuses change.

**`id`:** a stable handle, not an order; a retired id is never reused. The prefix is the section: `CARD` measurements on the card, `INP` inpainting, outpainting and edits, `GEN` generation, jobs and models, `PERF` speed, `GAL` gallery and storage, `PA` prompt analyser, `UI` the page, `DEV` docs and tooling. A review's own id (10-12 F7, U61) is named in the clause. One id was reused all the same: `GEN#23` named a wording fix (opened and closed 2026-10-25), then [More seeds' redesign](docs/plan-more-seeds-ux.md) (2026-10-26, closed with `UI#13`).

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
