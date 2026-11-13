# TODO: open work

The maintainer's working backlog for this personal project: it records what is planned, not a roadmap or a promise. An index of open work, one row per item; each row points to the doc that owns the detail. The rules for editing it are at the end, in [How to read this file](#how-to-read-this-file). A fresh thread starts here: read **NOW**, then the owner of the row it picks up.

## NOW

**The only ordering in this file.** Keep every `DECISION` and `USER` row here. Keep this block to the current aim and the next few actions; results go in their owners (`docs/verified.md` for what the card measured, the plan or design notes for what was built).

**Aim (2026-10-29):** keep useful work ready for Matt's common workstation tasks. [Reported use](docs/design/preferences.md) guides priority; it does not approve the candidate redesign.

**Ready backlog (updated 2026-11-13):** 15 `OPEN` rows from the [2026-11-13 TODO audit](docs/maintainer/review-2026-11-13-todo.md#ready-work-open), all CPU or browser work. `DEV#64` (the passcode renewing on a marked poll), `GAL#28` (Windows sidecar reads and saves) and `GAL#29` (a deleted PNG kept in bundles) finished the same day. Suggested order: `GEN#71` before Matt's `GEN#68` look, then the rest. `DEV#58` (PR #72) and `GEN#46` (FLUX 3 [Klein]'s real tokenizers in CI, PR #73) finished 2026-11-13. The [2026-11-12 TODO audit](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12)'s rows, with `GAL#27` and `UI#53` added at Matt's request, were built 2026-11-12 and 13: those finished were removed, and the rest wait on Matt (`GEN#66` and `GEN#68`, `USER`), on the card (`GEN#6`, `GEN#48`'s stage 2) or on his call (`INP#12`, deferred). Local CPU, stub-script and fake-browser checks do not establish card, clean-machine or real-device acceptance.

**Decisions:** no unanswered `DECISION` rows. On 2026-11-13 Matt accepted both of the [2026-11-13 audit](docs/maintainer/review-2026-11-13-todo.md#matts-answers-2026-11-13)'s (`GAL#31`, `UI#56`, now `OPEN`). On 2026-11-12 Matt accepted every recommended answer of the [2026-11-12 audit](docs/maintainer/review-2026-11-12-todo.md#matts-answers-2026-11-12): 11 rows became `OPEN`, `CARD#4`, `CARD#6` and `CARD#12` were retired under the K4/K5 Never ratings, and `GEN#15` stays a model-offload card render. Decisions in force that limit work: `GEN#39` and unattended backups (`GAL#22`, `GAL#6`) declined; `GEN#8`, `GEN#20` and `INP#10` deferred by Matt; `PERF#7`'s answers (cache beside the gallery, static shapes, shipping after `PERF#1`); `DEV#20`'s 4 GiB commit floor (a card session runs on a quiet machine); `GEN#45`'s FLUX 3 scope (Klein and Klein Base, the upsampler opt-in only); the passcode's flagged choices as written (2026-11-10); Animagine XL 4.0 as the first open checkpoint (`GEN#59`) and Anima for the first app-owned Refine (`GEN#48`), both 2026-11-11. The history is in the owners and in Review coverage below.

**Needs Matt's hands or card:** `DEV#39`'s deliberate ruleset edit before a hosted installer CI job can be added; `DEV#32`'s stage 4, the passcode on the real machines; `UI#40`'s phone/platform acceptance, workbench task trials (`DEV#1`, kit refreshed 2026-10-30), the typed Ctrl-C on the fake for `DEV#23`, `GEN#66`'s `powercfg /requests` check, `GEN#68`'s look at Models > On disk over the real cache, hands-on accessibility/desktop checks (`UI#7`), and every `CARD` row, including the speed qualifications `PERF#1`, `PERF#3` and `PERF#4`, and `INP#11`'s acceptance step (one load). `PERF#2`'s card measurement waits on `PERF#1`'s.

**GPU runs are Matt's to run or OK, never past the card's memory** (16 GB on the 5080, 8 GB on the 3070; every `CARD` row). Check the checkpoint's measured peaks in `docs/verified.md` first; a run that spills into shared memory proves only that it doesn't fit.

## Product design

**Owner:** [docs/design/README.md](docs/design/README.md); [validation](docs/design/validation.md) owns the remaining task trials and evidence. Tooling row `DEV#23`: the [card harness](docs/maintainer/plan-card-harness.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#23` | USER | LOW | S+card | The eight `tools/card/card10-*.json` packs and their test are built 2026-10-30; the six page packs pass on the fake. Left: Matt types Ctrl-C on the fake for the two Ctrl-C packs, then stage 2 on the card. [CARD#10 step packs](docs/maintainer/plan-card-harness.md#card10-step-packs). |
| `DEV#1` | USER | LOW | M | Run common-task trials with Matt, comparing the current app and candidate layouts before choosing redesign slices. [Protocol](docs/design/validation.md#one-user-formative-protocol). |

## First-run experience

**Owner:** [docs/maintainer/review-2026-11-01-first-run.md](docs/maintainer/review-2026-11-01-first-run.md): findings F1-F8, the stages in order, and what was not taken. The rows from `DEV#41` on: [docs/maintainer/review-2026-11-10-first-time.md](docs/maintainer/review-2026-11-10-first-time.md), a first-time user after the installer.

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#29` | DEFERRED | LOW | S | Stage 5: hand-cut version tags for `update.cmd` to follow, and a Pinokio script; paused by priority (the installer is built). [Stage 5](docs/maintainer/review-2026-11-01-first-run.md#stages). |

## Server, security and tooling

**Owner:** [docs/maintainer/review-2026-11-10-peers.md](docs/maintainer/review-2026-11-10-peers.md): findings S1-S7 and T1-T3, each with its first step and what was cut. `DEV#65`-`DEV#70`: the [2026-11-13 audit](docs/maintainer/review-2026-11-13-todo.md#ready-work-open).

| id | status | sev | size | item |
|---|---|---|---|---|
| `DEV#32` | USER | MED | S | The opt-in passcode (`AURADREAM_PASSCODE`) is built through stage 3, 2026-11-10. Left: stage 4 on the machines, Matt's: the Linux service from a phone and another PC, Safari's autofill, a session across a restart and an update, the scripts through the passcode, a wrong passcode's lock. [Build order](docs/design-notes.md#build-order); [S2](docs/maintainer/review-2026-11-10-peers.md#s2-passcode-beyond-loopback). |
| `DEV#65` | OPEN | LOW | S | smoke-compact-setup's reference reorder races the loras refresh after its klein switch: wait for the answer before focusing a chip. [DEV#65](docs/maintainer/review-2026-11-13-todo.md#dev65-smoke-compact-setups-reorder-races-the-loras-refresh). |
| `DEV#66` | OPEN | LOW | S | `linux/update.sh` never compares the installed systemd unit with `install-service.sh`'s (a unit from before `DEV#55` lacks exit 78's no-restart). [DEV#66](docs/maintainer/review-2026-11-13-todo.md#dev66-linuxupdatesh-never-checks-the-installed-unit). |
| `DEV#67` | OPEN | LOW | S | `bin\auradream.ps1` has no uv fallback and exits 0 when uv is missing; its `.cmd` has both. [DEV#67](docs/maintainer/review-2026-11-13-todo.md#dev67-binauradreamps1-lost-step-with-its-cmd). |
| `DEV#68` | OPEN | LOW | S | `update.ps1` tests a relative `AURADREAM_DB_PATH` from the caller's folder, so run elsewhere it skips the gallery copy. [DEV#68](docs/maintainer/review-2026-11-13-todo.md#dev68-updateps1-checks-a-relative-gallery-path-from-the-wrong-folder). |
| `DEV#69` | OPEN | LOW | S | `update.ps1` on a detached HEAD (after "Going back"): RemoteException noise lines and a generic stop instead of `git checkout main`. [DEV#69](docs/maintainer/review-2026-11-13-todo.md#dev69-updateps1-on-a-detached-head-noise-and-no-named-stop). |
| `DEV#70` | OPEN | LOW | S | `wsl/env.sh` sourced again keeps its marked Nunchaku default after the quantization or format changes. [DEV#70](docs/maintainer/review-2026-11-13-todo.md#dev70-wslenvsh-keeps-a-stale-default-when-sourced-again). |
| `DEV#39` | USER | LOW | M | Local shellcheck/PSScriptAnalyzer audit is built. Left: Matt's deliberate CI ruleset edit, then a hosted installer job up to the GPU check. [T2](docs/maintainer/review-2026-11-10-peers.md#t2-the-scripts-in-ci). |

## Gallery and storage

**Owner:** the [2026-11-13 audit](docs/maintainer/review-2026-11-13-todo.md).

| id | status | sev | size | item |
|---|---|---|---|---|
| `GAL#30` | OPEN | LOW | S | The Runs list opens a database session and validates every cell per listed run (about 1.5 s at 200 runs), again after every job. [GAL#30](docs/maintainer/review-2026-11-13-todo.md#gal30-the-runs-list-opens-one-database-session-per-run). |
| `GAL#31` | OPEN | LOW | S | The page's Delete unstarred accepts positive free words, prefix-matched as the gallery's search; the odd-quote fallback and `-word` stay refused, `clear --where` strict (Matt, 2026-11-13). [GAL#31](docs/maintainer/review-2026-11-13-todo.md#matts-answers-2026-11-13). |

## Measurements on the card

**Owner:** [docs/verified.md](docs/verified.md): each row's "Not measured" or "Not verified" list. A result is recorded there, with its date, in the same commit that removes the row here.

| id | status | sev | size | item |
|---|---|---|---|---|
| `CARD#1` | CARD | LOW | S+card | Inpaint at fp8: FLUX.1 dev and schnell at 0.95, a LoRA through a FLUX.1 inpaint, the seam by number on FLUX.1. Nunchaku is `CARD#3`, later families are `INP`. [Inpaint](docs/verified.md#inpaint). |
| `CARD#2` | CARD | LOW | S+card | Outpaint: FLUX.1 dev and schnell, and the outpaint's seam by number on FLUX.1. [Outpaint](docs/verified.md#outpaint). |
| `CARD#3` | CARD | LOW | S+card | An inpaint under WSL through the real server and the Inpaint dialog, and over the FLUX.1 dev and schnell lite transformers. [Under Nunchaku](docs/verified.md#under-nunchaku-wsl-2026-10-25). |
| `CARD#7` | CARD | LOW | M+card | Refine on FLUX.1 at fp8: dev, what schnell's starting strength keeps, true CFG through FluxImg2Img. Offload none and sequential past Z-Image Turbo. [Refine pass](docs/verified.md#refine-pass). |
| `CARD#9` | CARD | LOW | S+card | ERNIE-Image Turbo on fp8: seed reproduction across two processes, whether the 4 MP batch of two fits (the harness's projection refused it), and the page's Expand line and clock over the rewriter's real first-use download. [Behaviour](docs/verified.md#behaviour). |
| `CARD#10` | CARD | LOW | M+card | On the real server, fixes checked only on the fake: 9377de1, 7733f81, 6a7570d, ad01fdc, 2a4d2cc, 2b89ed5. R50's fix (1744dc3) with a typed Ctrl-C, and Ctrl-C to an idle server with a model loaded. [Behaviour](docs/verified.md#behaviour); [10-12 not reached](docs/maintainer/review-2026-10-12.md#review--2026-10-12). |
| `CARD#11` | CARD | LOW | M+card | The critic: a reorder that reaches the partial-placement check (none of 16 did; every 4B reorder changed words); FLUX.1 Krea [dev] at int8 or sequential beside it, and its rewrites against originals over more seeds; kill-on-close under WSL; Ctrl-C during a critique. [Prompt analyser](docs/verified.md#prompt-analyser). |
| `CARD#13` | CARD | LOW | M+card | Only the masked area: FLUX.1 fp8/model-offload crop on a 2.4-8.4 MP source (peak near 12.2/12.8 GiB) and its seams at padding 0/32/64 under model offload (sequential measured 2026-11-11); other Z-Image crop checkpoints. One load per session, with a lower desktop VRAM baseline. [Remaining measurements](docs/verified.md#only-the-masked-area). |
| `CARD#14` | CARD | MED | S+card | On the 3070 server: a suspend with a model loaded, and whether the loaded pipeline survives the resume (`NVreg_PreserveVideoMemoryAllocations=1` is set). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#15` | CARD | LOW | S+card | Nunchaku int4 on the 3070 (sm_86): whether the lite-infer kernels load and render, its time and peak. [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#16` | CARD | LOW | S+card | The prompt critic's pinned CUDA 13.4 llama.cpp build under the 3070's 13.2 driver (first use, a critique). [RTX 3070](docs/verified.md#rtx-3070-8-gb-native-ubuntu-2026-11-03). |
| `CARD#21` | DEFERRED | LOW | M+card | FLUX 3 on the 8 GB 3070 as lite-infer int4 (its index points at nvfp4, which sm_86 cannot run): wire the int4 load, then fit and time; the upsampler cannot fit. Only if Matt wants FLUX 3 there. [FLUX 3](docs/design-notes.md#flux-3-klein-an-app-owned-pipeline-over-pinned-folders). |

## Inpainting, outpainting and edits

**Owner:** [docs/maintainer/plan-inpainting.md](docs/maintainer/plan-inpainting.md#phases) and [docs/maintainer/plan-outpainting.md](docs/maintainer/plan-outpainting.md#phases), their Phases; what was built differently is in the design notes' [Inpainting](docs/design-notes.md#inpainting) and [Outpainting](docs/design-notes.md#outpainting). `INP#11` and `INP#12`: the [instruction-editing plan](docs/maintainer/plan-instruction-editing.md) and [detailer plan](docs/maintainer/plan-detailer.md), requested 2026-10-27.

| id | status | sev | size | item |
|---|---|---|---|---|
| `INP#11` | CARD | LOW | XL+card | Stages 0-3 done; stage 4's ten slices on main since PR #17 (Kontext listed, the real gate open inside the measured envelope: fp8, model offload, Windows, 1:1 at 512x512). Left: the card acceptance step, one load on the real server and page (the envelope re-measured on diffusers 0.42.1, 2026-11-10). [Card step](docs/maintainer/plan-instruction-editing.md#card-acceptance-step). |
| `INP#5` | DEFERRED | LOW | M+card | Phase 4, klein: one reference beside the source, its mu recipe, memory with the sequence doubled, `smoke-inpaint-klein`. Matt doesn't run klein. |
| `INP#6` | DEFERRED | LOW | M+card | Phase 5, Qwen-Image under Nunchaku (WSL), only if the inpaint class runs over the Nunchaku transformer at all. |
| `INP#10` | DEFERRED | LOW | M | Outpaint, later if wanted: the server's fill shown in the frame (kept for later, 2026-10-17), an outpaint over a painted mask, a second fill. |
| `INP#12` | DEFERRED | LOW | M | Stage 1's desk half done 2026-11-12: YuNet and MediaPipe Hands pass, both photo-trained; the anime detectors stop at AGPL provenance. A CPU trial needs Matt's call: photo-trained detectors, or a new bounded search. [Desk results](docs/maintainer/plan-detailer.md#stage-1-desk-results-2026-11-12). |
| `INP#13` | BLOCKED | LOW | L+card | After `INP#11`'s card acceptance: open the plan's frozen aspects under the 512² area, and a second frozen recipe at 1024x1024 1:1 with its own one-load card protocol. [R3](docs/maintainer/review-2026-11-10-peers.md#r3-a-wider-instruction-edit-envelope). |

## Generation, jobs and models

**Owner:** the plan or section each row links; `GEN#48`-`GEN#56`: the [2026-11-10 peer review](docs/maintainer/review-2026-11-10-peers.md)'s R1-R10. Otherwise: [peak memory](docs/maintainer/plan-peak-vram.md) and [Then refine](docs/maintainer/plan-then-refine.md) for the card rows; the design notes, [docs/verified.md](docs/verified.md) or a review for the deferred ones.

| id | status | sev | size | item |
|---|---|---|---|---|
| `GEN#15` | CARD | LOW | S+card | Peak memory per image, phase 3: a FLUX.1 Krea [dev] model-offload text-to-image render, its recorded peak pair beside `nvidia-smi` and shared memory (the 2026-11-11 sequential inpaint does not count, 2026-11-12). [Plan](docs/maintainer/plan-peak-vram.md). |
| `GEN#16` | CARD | LOW | S+card | Generate then refine: phase 3's FLUX.1 Krea dev night at 1.5x, peak of the alternation and the pair's time. [Plan](docs/maintainer/plan-then-refine.md). |
| `GEN#6` | CARD | LOW | S+card | SDXL map and the non-flow rule built on CPU 2026-11-12 (R² 0.909 held out). Left, at Matt's OK: one card glance at an Animagine render's live-preview frames (colour; is the noisy first frame acceptable). FLUX.2's map stays deferred. [Results](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12). |
| `GEN#8` | DEFERRED | LOW | L | U61 part B: one's own photo or sketch as a reference (import route, a source marker through storage, reindex and reruns; add, drop and paste). Matt deferred it 2026-10-05; only when he raises it. [References](docs/design-notes.md#reference-images-by-gallery-id). |
| `GEN#20` | DEFERRED | LOW | L+card | Opt-in A1111 weighting and 77-token chunking on SDXL, deferred by Matt 2026-10-27 because he does not write weighted prompts; revisit on request. [10-24 G10](docs/maintainer/review-2026-10-24.md#g10-a1111-style-weighting-and-long-prompts-on-sdxl). |
| `GEN#48` | CARD | MED | L+card | Stage 1 built 2026-11-12: private Anima Img2Img passes with CPU tests, wired to nothing. Next: stage 2, Matt's authorized card qualification of Turbo (then Aesthetic, Preview; one load each); stage 3 wires it into Refine. [Plan](docs/maintainer/plan-anima-refine.md). |
| `GEN#49` | CARD | LOW | L+card | Tiled refine past the one-pass size guards. First a throwaway card look, with Matt's OK: one 2x source as four overlapping masked-area regions at strength 0.3-0.4; a plan only if the seams pass. [R2](docs/maintainer/review-2026-11-10-peers.md#r2-tiled-refine). |
| `GEN#66` | USER | MED | S | Built 2026-11-12: under WSL a Windows power request through interop, or an honest "nothing held". Left: Matt runs `powercfg /requests` (elevated) with a busy WSL server, sees "AuraDream is in use", and that it is gone once idle or stopped. [Results](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12). |
| `GEN#69` | OPEN | LOW | S | Grid refusals name schema fields: a variable block over 2,000 characters ("prompt_axis: ...") and `nan` in a CFG or Mu axis. [GEN#69](docs/maintainer/review-2026-11-13-todo.md#gen69-grid-refusals-name-schema-fields). |
| `GEN#70` | OPEN | LOW | S | Models' Speed facts still show 10-05's busy first runs; take in the warm fp8 repeats the card recorded 2026-10-25. [GEN#70](docs/maintainer/review-2026-11-13-todo.md#gen70-models-speed-facts-miss-the-warm-fp8-repeats). |
| `GEN#71` | OPEN | LOW | S | Models > On disk counts a checkpoint cached at an unrecorded commit as 0 bytes under Other repositories; Unknown, as Models says. Before `GEN#68`'s look. [GEN#71](docs/maintainer/review-2026-11-13-todo.md#gen71-models--on-disk-shows-an-unrecorded-revision-as-0-bytes). |
| `GEN#68` | USER | LOW | S | Models > On disk is built, read-only (2026-11-12). Left: Matt looks at it over his real cache; only then is the guarded delete designed, as a new row. [Results](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12); [R9](docs/maintainer/review-2026-11-10-peers.md#r9-storage-view). |

## Speed

**Owner:** the [2026-10-25 review](docs/maintainer/review-2026-10-25.md#speed), including each row's CPU-only next planning slice. The measured weight-only fp8/int8 cases run slower a step than bf16; none of the proposed speed tools is used yet. Each pixel-changing choice is recorded for reproduction.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PERF#1` | CARD | MED | L+card | Qualify toolchain/cache and the card; source, bounded diagnostics presentation and Windows browser smokes are done. Public nonempty admission stays closed. [Plan](docs/maintainer/plan-regional-compilation.md). |
| `PERF#3` | CARD | LOW | M+card | Qualify Native/cuDNN speed, fit and repeatability. Public nonempty admission remains closed. [Plan](docs/maintainer/plan-attention-backends.md). |
| `PERF#4` | CARD | LOW | L+card | Run the fixed off/on quality, time and memory comparisons, then judge the pairs. Public caching remains closed. [Plan](docs/maintainer/plan-step-caching.md). |
| `PERF#2` | BLOCKED | LOW | M+card | The card measurement of a quantization choice with fp8 activations (torchao `Float8DynamicActivationFloat8WeightConfig`); waits on `PERF#1`'s card qualification. Its CPU plan is done (`fp8-dynamic`, PerTensor, compared against fp8 only); the CPU source slice and card run are its stages 2-3. [Plan](docs/maintainer/plan-fp8-activations.md); [10-25 R2](docs/maintainer/review-2026-10-25.md#r2-fp8-with-fp8-activations). |
| `PERF#7` | BLOCKED | LOW | — | Whether the compiled mode ships waits on `PERF#1`'s card table; Matt leans to an opt-in Models choice with Eager the default if it pays off. Cache location, size policy and static shapes are decided. [Questions](docs/maintainer/plan-regional-compilation.md#matts-questions). |

## Prompt analyser

**Owner:** the design notes' [prompt analyser](docs/design-notes.md#the-prompt-analyser) for the deferred item; its [open questions](docs/design-notes.md#open-questions-from-the-analyser) for decisions.

| id | status | sev | size | item |
|---|---|---|---|---|
| `PA#17` | OPEN | LOW | S | Tag verdicts call a Danbooru alias (`twin tails`, `blond hair`) a tag with no posts; resolve the alias table and name the canonical tag. [PA#17](docs/maintainer/review-2026-11-13-todo.md#pa17-tag-verdicts-miss-danbooru-aliases). |
| `PA#9` | DEFERRED | LOW | L | Scoring comparison images automatically with a VLM (does each element appear); accuracy unmeasured. [Prompt analyser](docs/design-notes.md#the-prompt-analyser). |

## The page

**Owner:** [docs/design/validation.md](docs/design/validation.md) for the manual checks; [the phone review](docs/maintainer/review-2026-11-05-phone.md) for the phone redesign; the [2026-11-13 audit](docs/maintainer/review-2026-11-13-todo.md) for `UI#54`-`UI#56`.

| id | status | sev | size | item |
|---|---|---|---|---|
| `UI#54` | OPEN | LOW | S | The offline page's live region speaks every 10 s while the PC sleeps; its Try again and passcode's Sign in are held with no reason. [UI#54](docs/maintainer/review-2026-11-13-todo.md#ui54-the-offline-page-speaks-every-10-s-held-buttons-give-no-reason). |
| `UI#55` | OPEN | LOW | S | Notify when done unticks itself on Android Chrome (an illegal `Notification` constructor); show it through the service worker. [UI#55](docs/maintainer/review-2026-11-13-todo.md#ui55-notify-when-done-on-a-phone-through-the-service-worker). |
| `UI#56` | OPEN | LOW | S | U runs Reuse settings in the lightbox and on a focused card or sheet cell, scoped like M, with help rows (Matt, 2026-11-13). [UI#56](docs/maintainer/review-2026-11-13-todo.md#matts-answers-2026-11-13). |
| `UI#40` | USER | MED | M | Phone acceptance: physical iPhone keyboard/sharing/gestures, Android/system and native iOS edge Back, landscape instruction caret and actual iPad Split View. [Phone review](docs/maintainer/review-2026-11-05-phone.md#10-edit-dialog-keyboard-acceptance-2026-11-07). |
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
| 2026-10-25 | R1-R10 indexed above. Tooling has [execution stages](docs/maintainer/plan-tooling.md); speed/backups have CPU-only planning starts. Matt requested both edit plans 2026-10-27; `INP#11` is in progress and detailer implementation remains unscheduled. |
| 2026-10-26 workflow | Three reproduced findings fixed and qualified: mixed prompt prose (`PA#10`), incomplete numbers (`GEN#24`) and unseen queued outcomes (`GEN#25`). [Evidence and results](docs/maintainer/review-2026-10-26-ux.md). |
| 2026-10-26 keyboard | All three approved stages (`UI#17`-`UI#19`) fixed and qualified, including fast routes and compact setup. [Evidence](docs/maintainer/plan-shortcut-workflow.md). |
| 2026-10-26 sizing/Bundles | Password association (`UI#14`), clear-count explanation (`UI#15`) and frequent secondary target sizing (`UI#16`) fixed and qualified. [Evidence and results](docs/maintainer/review-2026-10-26-sizing-bundles.md). |
| 2026-10-26 TODO audit | [Owner/code reconciliation](docs/maintainer/review-2026-10-26-todo.md): Marian chunk truncation (`GEN#26`), portable screenshots (`DEV#7`), restored critic decision (`PA#11`) and bounded tooling/planning stages. |
| 2026-10-26 overnight | `GEN#26`, `DEV#5`-`DEV#7` built and locally qualified; all four feature plans written. [Tooling evidence](docs/maintainer/plan-tooling.md). Linux browser qualification (`DEV#4`) closed with PR #15 (hosted, 77 of 77); no GPU acceptance is claimed. |
| 2026-10-27 decisions | Matt deferred `GEN#20`; the requested [instruction-editing](docs/maintainer/plan-instruction-editing.md) and [detailer](docs/maintainer/plan-detailer.md) plans are written. Matt then commissioned starting `INP#11` without the card; `INP#12` remains deferred. `PA#11` closed by keeping critic rewrites available with the existing [omission warning](docs/design-notes.md#open-questions-from-the-analyser). |
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
| 2026-11-10 peers | [The app against comparable projects](docs/maintainer/review-2026-11-10-peers.md): 48 candidates checked against the code and earlier decisions, 23 kept (`GEN#48`-`GEN#56`, `INP#13`, `DEV#31`-`DEV#40`, `UI#41`-`UI#43`), the rest recorded as not taken; `DEV#31`, `DEV#32` (through stage 3), `DEV#34`-`DEV#38`, `DEV#40`, `GEN#51`-`GEN#54`, `GEN#56`, `UI#42` and `GEN#57` (opened and built the same day) built 2026-11-10; `GEN#50`, `GEN#55`'s first slice, `DEV#33`, `UI#41`, `UI#43` and `DEV#39`'s local audit built 2026-11-11; `GEN#48`, `GEN#49`, `INP#13`, `DEV#32` stage 4 and `DEV#39`'s hosted job indexed above. |
| 2026-11-10 first time | [A first-time user, after the installer](docs/maintainer/review-2026-11-10-first-time.md): 56 findings checked by skeptics, one refuted; merged to 28, 27 as rows (`DEV#41`-`DEV#53`, `GEN#58`-`GEN#62`, `UI#44`-`UI#50`, `PA#13`, `PA#14`), all built 2026-11-10/11; F27 closed with `GEN#65`'s shard maps (2026-11-12). |
| 2026-11-12 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-11-12-todo.md): stale NOW and coverage text fixed, `DEV#20` stage 2 recorded; 26 verified proposals added as 14 `OPEN` rows and 13 `DECISION` rows; Matt accepted every recommendation the same day (11 more `OPEN`, `CARD#4`/`CARD#6`/`CARD#12` retired). Built 2026-11-12/13 with `GAL#27` and `UI#53` (PR #71, [results](docs/maintainer/review-2026-11-12-todo.md#results-2026-11-12)), `DEV#58` last (PR #72, every smoke polling its expects); what is left of `GEN#6`, `GEN#48`, `GEN#66`, `GEN#68` and `INP#12` is indexed above. `GEN#46` closed 2026-11-13 (PR #73): CI's account reads both FLUX 3 licences, and FLUX 3 and Klein Base are `REAL` in the encoding tests and the CI tokenizer fetch. |
| 2026-11-13 TODO audit | [Reconciliation and new work](docs/maintainer/review-2026-11-13-todo.md): coverage table, `GEN#15`'s owner and two F27 references fixed; two rounds of lenses, 27 proposals, 4 refuted; 16 `OPEN` rows and 2 `DECISION` rows added, both accepted by Matt the same day (`GAL#31`, `UI#56`, `OPEN`); `DEV#64`, `GAL#28` and `GAL#29` built the same day ([results](docs/maintainer/review-2026-11-13-todo.md#results)). |

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
