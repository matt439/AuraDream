# Overnight TODO work, 2026-11-16 to 17

Matt authorized autonomous TODO work while this workstation was free overnight.
Work stayed on `work/overnight-todo-2026-11-16`, with local commits only.
No push or pull request was authorized. Owners and dated verified results
remain authoritative; this is the handoff between the completed slices.

## Completed releases and fixes

- `GEN#15`: five real FLUX.1 Krea dev fp8/model direct renders agreed across
  allocator, backend, job/API, SQL and PNG peak pairs. The row was removed.
  [Receipt](flux-memory-2026-11-16.json).
- `GEN#48`: bounded Refine for Anima Turbo, Aesthetic and 2.9B Preview is
  implemented, independently measured and integrated. Private trials,
  nonempty-negative probes where applicable, public API/CLI-via-server and
  ordinary standalone CLI sessions passed separately. Four fake-browser
  rows passed. The row was removed in `097b498`.
  [Owner and limits](plan-anima-refine.md).
- A real Windows CLI failure was fixed: the old two-second server probe
  expired just before Windows delivered closed-port refusal. The bounded
  default is five seconds; busy/unknown/timeout/passcode refusals remain.
  The original failed standalone attempt is retained. Commit `90eb8c5`.
- The pinned type-check wrapper now rejects incomplete/ignored include
  scopes, including a reproduced absolute-path false pass. The measured
  source count must match the checker. Commit `42096d9`.
- The private Anima card runner now uses immutable acknowledgements and
  bounded Windows lock retries, with real locked-reader regressions. The
  original IPC-failed trial remains failed, followed by a fresh full run.

The Anima release is exact Windows/RTX 5080 CUDA 0, fp8/model offload,
Eager/Native, diffusers 0.42.1 and 1024-square RGB source/output. Turbo is
8 steps/CFG 1, opening 0.25; Aesthetic/Preview are 30 steps/CFG 4, opening
0.125. Measured strengths are 0.125/0.25/0.4. No enlargement, other loader,
hardware/platform or general image-improvement claim follows.

## Partial measurements and honest limits

- `GEN#6`: fourteen real SDXL callback/SSE preview JPEGs matched; retained
  representative frames show noise developing into the final composition.
  The row is now `USER` for Matt's early-frame preference and real-page look.
  [Capture](gen6-preview-2026-11-17.json).
- `CARD#3`: WSL Z-Image Turbo's real server/dialog, compositing, source/mask,
  storage, cancellation and recovery checks passed. All 842052 unmasked
  pixels stayed unchanged. Nunchaku repeated pixels differed, as recorded.
  The original overall exit failed only on oversized evidence export;
  a separate CPU-only export succeeded, preserving that failure. FLUX dev
  and schnell lite remain. [Receipt](card3-z-turbo-wsl-2026-11-17.json).
- `PERF#1`: cached-overlay Triton and Inductor executed tiny real CUDA work
  with exact results and strict recompile policy. Explicit bundled compiler
  paths were selected and verified for the UV overlay. Base dependencies
  were unchanged;
  this is not application/model/performance/public compiler qualification.
  [Receipt](perf1-tiny-toolchain-2026-11-17.json).
- Private attention/FBC and regional registration were checked against
  exact diffusers 0.42.1 on CPU with actual tiny installed classes. Their
  nonempty public execution support remains held; no large-model speed or
  memory qualification follows from those checks.
- `CARD#11`: three new fixed partial-placement critic responses changed
  words and were correctly rejected before the target branch. A fully
  placed control retained `check`. All process/listener/gate and memory
  checks passed; the target remains open. No output was repaired or retried,
  and numerical GPU layer counts were absent.
  [Receipt](card11-critic-2026-11-17.json).
- `GEN#49`: the four-region SDXL seam look passed only its planning gate.
  No obvious tile outline/cross was seen by two reviewers; substantial
  redraw and possible added detail mean fidelity/useful gain are unproven.
  The [plan](plan-tiled-refine.md) owns private orchestration and the remaining
  durable-intent/backend/card/public release work.
  [Exploratory receipt](gen49-sdxl-four-region-2026-11-17.json).
  Its private CPU prototype is now built and tested; no production import,
  public admission or actual pipeline adapter was added. The row is `OPEN`
  for that larger integration, not closed by the experiment or prototype.

## Resource stops and remaining authorization

The first `GEN#16` 1536-square FLUX Refine reached 15632 MiB whole-card
before one completed denoising step, above the unchanged 15500 MiB limit.
It was stopped without a saved result. No alternation or timing comparison
was qualified. The successful preceding direct session also reached only
4.03 GiB commit headroom at its narrowest sample, so this is not spare
capacity for another large load.

The 06:28 Brisbane read-only snapshot had 34.32 GiB available RAM and
30.25 GiB commit headroom, with 1842 MiB idle card. Larger Windows protocols
require their recorded 40/45 GiB
RAM and 52 GiB commit admission; WSL dev/schnell requires 36/36 GiB and a
24 GiB Linux RSS limit. These admissions were unavailable and were not
lowered. Live card/shared and 4 GiB RAM/commit floors stayed unchanged.
No user applications were closed to force a fit.

The prepared scratch protocols remain under `work/`: `perf3_card_private.py`,
`perf4_card_private.py`, `instruction-acceptance/` and `card3_flux_trial.py`.
The latter's frozen readiness pack is
`work/card3-flux-prepared-readiness-90eb8c5-20261117-a1/`; its runtime was
reviewed through `90eb8c5`, not any later source change. Recheck source
identity, ownership, actual current headroom and the owner before a run.
Every actual attempt needs a new evidence directory. Prepared is not run.

`PA#18`'s explicit assistant GPU hold remains. `INP#12` still needs
Matt-picked sources. Real-cache deletion, physical-device trials, typed
Ctrl-C, external 3070 operations, declined/deferred work and other hands-on
acceptance retain their own gates. The critic-only run did not exercise
the assistant.

## Validation and final state

Before the private tiled prototype, the integrated suite passed **11071
tests, 325 skipped**, 85 warnings, exit 0 in 356.71 seconds, with
`CUDA_VISIBLE_DEVICES=-1`. Ruff passed and pinned Pyright checked 217 sources
with zero diagnostics. The first full attempt's sole failure was a rules
section growing past its frozen bound; whole-bullet topic headings fixed
it before the successful fresh full run. Failed logs remain preserved.

The final private tiled source passed **11167 tests, 325 skipped**, 85
warnings, exit 0 in 418.69 seconds (`work/overnight-tiled-final-2026-11-17-cpu.log`).
Its focused gate passed 110 tests; Ruff passed and pinned Pyright checked
all 218 sources with zero errors/warnings. Ownership, exact measured mask
hashes/crop helpers, seed wrap, serial carry, later-tile cancellation/errors
and cold import order are covered. The actual saved-tile CPU replay also
passed: all four full intermediates and the final RGB match the measured
chain exactly, with 36 synthetic callbacks and four observed calls to the
real CPU overlay helper. The independently audited
[CPU replay receipt](tiled-refine-raw-cpu-replay-2026-11-17.json) keeps that
scope separate. It is not another GPU measurement. Two preceding
import-guard failures remain failed and preserved; CPU metadata imports
were distinguished from compiler execution before the fresh passing replay.

Finalization verified that the card/model trial runners had exited and ports
7992, 7993 and 7996 were bindable. It stopped only the identity-verified
overnight idle-sleep hold (PID 41908). A short scoped power request covers
the final local commit command and releases through its context manager;
the observed release and clean-tree result are retained in
`work/overnight-final-cleanup-2026-11-17.json`. No push or PR was made.

Raw images, SQLite/SSE/allocator/counter logs,
trial scripts and exact source snapshots remain in ignored `work/` folders;
tracked derivative receipts identify them and retain hashes. Keep those raw
folders for later inspection. No background schedule or backup was installed.
