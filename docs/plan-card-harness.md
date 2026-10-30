# Plan: a checked-in card-session harness

**Status:** `DEV#20` (card guard), `DEV#22` (image metrics) and `DEV#23`
(`CARD#10` step packs) opened 2026-10-29 from that day's TODO audit, which
Matt approved in full. `DEV#22` is built (2026-10-30, below); `DEV#20`'s
stage 1 is built (2026-10-30, below) and its card stage is open;
`DEV#23`'s first stage is built too (2026-10-30, below), but for its
hand-typed Ctrl-C runs on the fake. Each item's first stage is CPU,
fake-server or browser work; `DEV#20` and `DEV#23` each have a second stage
that needs the card and Matt's OK. None of them closes a `CARD`, `PERF` or
`GEN` row: the harness only makes those runs shorter and the same each time.

Every statement below about past card sessions was read on 2026-10-29 from
`docs/verified.md` and from the scripts still kept in
`%TEMP%/local-diffusion-card-20261025/` ("the cards directory", as
verified.md names it) and `%TEMP%/local-diffusion-inpaint-card/`. Those
directories are not in the repository and can be deleted at any time.
`%TEMP%/local-diffusion-page-gpu/`, which held the 2026-10-05 page checks'
`harness.py` and step files (verified.md, Behaviour), is already gone. So
the tests below copy the recorded numbers into the test files. They never
read `%TEMP%`.

## The problem

Every card session so far has written its own guards and drivers in
`%TEMP%`, and verified.md describes them only in prose. Among them:
`card4.py`, `card6.py`, `card_refine.py`, `card_wai.py`, `ernie_card.py` and
`monitor.py`, `flux1_dev_card.py`, `schnell_card.py`, `flux-krea/common.py`
and `steps.py`, `card11-critic/gpumon.py`, `card3_inpaint.py` with its
Windows-side `watch_card.ps1`, and the inpaint card's `measure.py`,
`measure_flux.py` and `measure_light.py`. `ls tools` shows no card tooling. The guards agree on
what to watch but differ on the limits (next section), and each session
re-derives them. The image metrics the tables report (mean difference,
PSNR, % of pixels off by more than 16, largest difference outside the mask,
the seam step) come from `card4-quant/compare.py` and the inpaint card's
`seam_insert.py`, and neither is checked in. The `CARD#10` recheck depends on
browser step files that were never committed, and the 2026-10-05 copies are
lost.

The memory rules that apply to every card run: a run must not spill past the
16 GB card ("GPU runs must not spill": stop a ladder at the size that fills
the card, at about 15.5 GB whole-card, and run it one size at a time in the
foreground); and a card OK is not a RAM OK ("GPU OK is not RAM OK": state the
run's system-RAM cost, load once per session, check commit headroom first).
The guard below enforces those limits. It does not replace asking Matt.

## What the past guards did

Read from the scripts' own docstrings and constants. Their whole-card and
free-RAM comparisons are strict (`used > limit`, `available < floor`), so a
reading equal to a limit does not trip:

| Guard | Session (script) | Limit |
| --- | --- | --- |
| Whole card (`nvidia-smi memory.used`) | ERNIE (`monitor.py`), Z-Image/Qwen refine (`card_refine.py`), WAI (`card_wai.py`), inpaint 2026-10-17/20/24 | 15,500 MiB |
| Whole card | FLUX.1 Krea dev server session (`flux-krea/common.py`) | 15,950 MiB ("the 4.2 MP FLUX spill read 15,918-15,922") |
| Whole card | `card4.py`; FLUX.1 inpaint before enabling (`measure_flux.py`) | 16,000 MiB (the card's total is 16,303) |
| Adapter shared GPU memory over its baseline | ERNIE (`typeperf "\GPU Adapter Memory(*)\Shared Usage"`, the adapter with the largest dedicated use); fixed-loader and Kontext runs 2026-10-27 | +1,024 MiB / "a 1 GiB rise" |
| Process shared GPU memory over its baseline | `card_refine.py`, `card_wai.py` | +512 MiB (adapter +1,024 beside it) |
| Process shared, card-conditioned | `card4.py` | +512 MiB while the card is at least 14,500 MiB, or +2,048 MiB whatever the card reads |
| Adapter shared, card-conditioned | `flux-krea/common.py` | +512 MiB over the loaded-idle baseline while the card is at least 15,300 MiB |
| Adapter shared over baseline (Windows watcher for a WSL run) | `card3-wsl/watch_card.ps1`, beside `card3_inpaint.py` | +300 MiB |
| Free RAM | most sessions | 4 GiB available; `measure_light.py` 3 GB |
| Commit headroom | `card4.py`, `card_refine.py` | 4 GiB (`CommitLimit - CommitTotal`, through `GetPerformanceInfo`) |
| Linear projection before a rung | `ernie_card.py` | refuse when the line through the last two measured sizes projects past 15,200 MiB, when an earlier rung peaked at 15,000 MiB or more, or after any earlier trip |
| Preflight | `ernie_card.py` / `flux-krea/common.py` / `card_wai.py` | card already above 6,500 / 3,000 / 3,500 MiB; less than 24 / 44 / 16 GiB of RAM available |
| Slow steps (the spill's sign) | `measure_flux.py`, `flux-krea/common.py` | two consecutive steps over 3 s, or 2.5 times the cell's own pace and at least 3 s slower |

What they did on a trip: a RAM trip wrote a `ram_guard` row
(`available_gib`, and `tree_gib`, `peak_rss_gib` or `commit_headroom_gib`
where the script had them), killed the watched server's process tree
(`psutil` children, recursive, killed in reverse) and exited 4. A card or
shared trip cancelled the render at its next step and refused every later
rung, since "a spill proves only 'does not fit'" (`ernie_card.py`).
`card11-critic/gpumon.py` printed a `TRIP` JSON line. `card4.py`'s exit codes were
0 ok, 1 load or render error, 3 spill or watchdog stop, 4 RAM guard,
5 skipped by preflight, 6 time cap, 7 card busy with another process.

## Card guard

`DEV#20`. A new file, `tools/card_guard.py`, with two halves.

**The decision** is pure: no I/O, no clock, no subprocess, and no import
of `backend`, torch or diffusers. Given a sample (whole-card MiB, adapter
and process shared MiB, available RAM, commit headroom), the session's
baseline and its limits, it answers one of `ok`, `trip` (with the limit that
was crossed and the reading) or, for a planned rung, `refuse-projected`
(with the projected MiB and the measured points). The projection is the one
`ernie_card.py` used. The x-axis is megapixels times batch, rounded to two
places (`round(w * h * batch / 1e6, 2)`), and the line runs through the last
two measured points. The decision also refuses a rung when an earlier peak
reached the refuse-at figure, or after any trip in the session.

**The sampler and the watcher** read the counters and act on the answer:
`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`
for the whole card, `typeperf` for the Windows GPU Adapter Memory and GPU
Process Memory shared counters, `psutil.virtual_memory().available` for
free RAM, and `GetPerformanceInfo` for commit headroom. `psutil` and
`numpy` are already installed as dependencies of `accelerate` (`uv.lock`).
The app does not declare them, and this item adds no dependency; declaring
them in the dev group would be a separate lock change. With `--watch-pid PID`
the watcher re-reads the process tree at every sample. On a trip it writes
the row, kills the tree (children in reverse order, then the root) and exits
with `card4.py`'s code: 3 for card or shared, 4 for RAM or commit, 5 for a
refused rung or preflight, 7 when the card is busy before the start. It
never starts, loads or cancels anything through the app's API. An in-process
harness can import the decision and call its own `should_cancel`.

**Limits in one block.** A frozen dataclass at the top of the file holds
the defaults. Each default is commented with the verified.md section it
came from, and each can be overridden for one session on the command line
or from a JSON file; the limits in force are written into the first row. The defaults are the
values most sessions used: card 15,500 MiB; projection limit 15,200 MiB;
refuse-at peak 15,000 MiB; adapter shared +1,024 MiB; process shared
+512 MiB; free RAM 4 GiB; commit headroom 4 GiB. There is no default for the
preflight RAM minimum. A load run must state one (`--need-ram-gib`), because
the RAM-at-the-load figures depend on the checkpoint (FLUX.1 fp8 needed
about 46 GiB of headroom before the fixed loader; the fixed loader peaked
at 27.9 GiB working set; SDXL at fp8 needs about 9.5 GB) and the memory rule
says to state that cost first. The guard never lowers a limit by itself.
Part of this item: `docs/rules.md` gets a Layout entry for the new file
(its `tools/` list) and a Tests sentence naming the block, so a later session
changes the defaults there and nowhere else. Both exist (2026-10-30).

**Rows.** JSONL, one object per line with wall time. Where the old
`results.jsonl` and `harness.jsonl` files had an event name it is kept
(`baseline`, `ram_guard`, `rung_refused`, `summary`; `TRIP` as `gpumon.py`
printed it), so old and new rows read alike; `sample` is new. The events:
`baseline`, `sample` (one every few seconds, plus every sample within
a margin of a limit), `ram_guard` (with `available_gib`,
`commit_headroom_gib`, `tree_gib`), `TRIP` (with `why`, the reading and
the limit), `rung_refused` (with `reason` as `"projected N MiB > 15200"`
and `measured`), and a closing `summary` (peaks, minimum free RAM, the
largest shared rise).

Constraints, from `AGENTS.md` and `docs/rules.md`:

- Tests run with no GPU, no weights and a throwaway database. The decision
  tests use fixed samples. Nothing races a clock, since CI's runners are
  slow; the watcher's loop takes an injected sampler and an injected sleep.
- The pinned Pyright gate and ruff cover `tools/`, so the module passes
  both. Windows-only calls (`typeperf`, `GetPerformanceInfo`) sit behind
  the sampler interface, so Linux CI imports the module and runs the tests.
- "GPU behaviour, pixels, timing and memory are the card's to say": no CPU
  or fake run of the guard is reported as a card measurement.

Non-goals: a step-pace guard (it needs the job's step stream, so it stays
with each session's driver); watching a WSL process tree (WSL runs keep a
Windows-side watcher for the card and shared counters, as on 2026-10-25,
plus the harness's own WSL RAM guard); any app or `backend/` change; a UI;
a CI job that touches a GPU.

### Card guard stages and completion criteria

1. **CPU (done 2026-10-30).** `tools/card_guard.py` and `tests/test_card_guard.py`,
   with the subprocess and counter readers stubbed. Fixtures copied from the
   recorded rows:
   - The 2026-10-25 RAM stops each trip on free RAM: 3.83, 3.93, 3.7, 3.8,
     3.78 and 3.7 GiB (verified.md, "RAM at the load (2026-10-25)"; the
     `ram_guard` rows in `flux-krea/`, `flux1-dev/`, `flux1-schnell/` and
     `zimage-qwen/`, and `card4-quant/results.jsonl`'s `ram_min_avail_gib`
     for 3.93 and 3.78). The Qwen-Image row trips on RAM, not commit
     (headroom 10.0 GiB). klein 9B's stop (`card6-catalogue/`) is recorded as
     `{"event": "stop", "why": "ram_guard", "available_gib": 4.0}`, a value
     rounded after the trip, so it is not used as a boundary fixture.
   - The fixed loader's lowest free RAM, 14.2 GiB, is `ok`.
   - The ERNIE gates: the points `[[1.05, 9815], [2.07, 10247]]` project
     11,145 MiB at 4.19 and answer `ok`; `[[1.05, 9815], [2.07, 10247],
     [4.19, 12631]]` (the line through the last two) project 17,354 MiB at
     8.39 and answer `refuse-projected`, matching the recorded `rung_refused`
     row's `measured` list and its reason `"projected 17354 MiB > 15200"`
     (`ernie/run-card/results.jsonl`). The 1 MP point's x is the 1.05 that
     `ernie_card.py` hard-codes.
   - Shared readings that must not trip: the first render's constant
     82-86 MiB, WAI's +8 MiB process and +82 MiB adapter rises
     (verified.md, "Peak per image and Then refine, WAI v17"), and the
     critic's own +172-180 MiB adapter allocation. A synthetic +1,100 MiB adapter rise trips.
   - A card reading of 15,501 MiB trips, and 15,500 does not.
   - Per-session overrides change the answer, and the defaults block is the
     only source of the defaults.
   - `docs/rules.md` has the Layout entry and the Tests sentence above.

   Then a local dry run: `tools/fake_server.py` on a free port with a fresh
   `--data`, watched with `--watch-pid` for a few minutes, never trips and
   writes `baseline`, `sample` and `summary` rows. Run again with
   `--ram-floor-gib` set above the RAM currently available, it trips at once,
   writes `ram_guard`, leaves no fake process and exits 4. That checks the
   kill path without using any RAM. Reading the counters starts no GPU work.
   ruff, the Pyright gate and the full pytest suite pass.
2. **+card, with Matt's OK.** One light session that was already going to
   run, whose RAM cost is stated beforehand (an SDXL checkpoint at fp8: the
   Animagine XL 4.0 LoRA row in verified.md's Behaviour held 9.4 GB of RAM
   at its peak and 6,421-6,664 MiB of the card), with the guard watching
   beside that session's own watcher. Their readings agree within one
   sample period, nothing trips, and nothing is pushed toward a spill or the
   RAM floor on purpose. verified.md's row for that session names the guard
   and its limits.

**Stage 1 built (2026-10-30).** `tools/card_guard.py watch [--watch-pid
PID] [--out rows.jsonl] [--period 1] [--row-every 5] [--duration S]` and
`plan --points JSON (--x X | --size WxH [--batch N]) [--tripped]`, each with
the limit flags (`--card-mib`, `--project-mib`, `--refuse-at-mib`,
`--adapter-shared-mib`, `--process-shared-mib`, `--ram-floor-gib`,
`--commit-floor-gib`, `--need-ram-gib`, `--busy-card-mib`, `--limits
FILE.json`). `decide`, `plan_rung` and `rung_x` are the pure decision;
`LocalHost` is the real sampler (the adapter with the largest dedicated use,
as ERNIE's guard chose it; the tree's GPU Process Memory shared counters on
that adapter, summed over its pids). `tests/test_card_guard.py` holds every
fixture above: the six RAM stops trip on RAM (the Qwen-Image one with
commit headroom 10.0), 14.2 GiB is `ok`, ERNIE's two gates answer 11,145
`ok` and `"projected 17354 MiB > 15200"` with the recorded `measured` list,
82-86, +8/+82 and +172-180 MiB do not trip and +1,100 does, 15,501 trips
and 15,500 does not, overrides from a flag and a file change the answer,
and every flag defaults to None so `LIMITS` is the only source. The watcher
runs there on a stub host with an injected sleep and clock.

Checked: the targeted tests (41), the full suite, ruff and the Pyright
gate. The dry run against `tools/fake_server.py` (port 8100, a fresh
`--data`, the venv's launcher with its two children as the watched tree)
read the counters only. With the defaults it tripped three times, each on
**commit headroom** (3.61, 3.16 and, with `--commit-floor-gib 1`, 0.51 GiB),
writing `ram_guard`, killing the fake's tree and exiting 4: other sessions'
test runs had the machine's commit near its limit that night (the OS's
own `\Memory\Commit Limit - Committed Bytes` read the same within about
0.3 GiB). So the quiet run was made with `--commit-floor-gib 0` for that
session only: 180 s, `baseline`, 30 `sample` rows and `summary`, exit 0,
the fake left running. Then `--ram-floor-gib 1000` against that fake
tripped at once (`ram_guard` with `available_gib` 15.91), left no fake
process and no listener on the port, and exited 4. None of these readings
is a card measurement.

Choices the plan left open, taken while Matt was away (2026-10-30),
revisit if wanted: the decision checks RAM, then commit, then the card,
then the adapter and process rises, so one sample answers one trip; a
rung refused after a trip or an earlier peak is also `refuse-projected`,
its `reason` saying which; the last two points at the same x refuse the
input (exit 2) rather than guess; preflight and the busy card write a
`rung_refused` row (exit 5 / 7) and kill nothing; "busy" is a whole-card
reading at the start above `--busy-card-mib`, which like `--need-ram-gib`
has no default; the watcher refuses (exit 2) a missing `--watch-pid` or a
missing nvidia-smi reading rather than watch blind; a `sample` row is
written every `--row-every` seconds and at every sample within 500 MiB of
a card or shared limit or 1 GiB of a RAM or commit floor (the margins sit
in the same block); the watched tree exiting ends the watch with exit 0.
Left: stage 2 (the card, with Matt's OK). The commit floor's trips on a
busy machine are worth Matt's look before that session: 4 GiB of commit
headroom was not there with several sessions' test runs resident.

**Hardened after review (2026-10-31): the card unread mid-watch.** The
watcher refused to start without a whole-card reading, but after the
first sample a sample with no card reading (nvidia-smi failing or timing
out) only skipped the card check, so it could watch blind for the rest of
the session. Now it counts such samples in a row: the first writes a
`card_unread` row (the other counters' fields), and the
`card_unread_samples`-th in a row (3 by default, in the `Limits` block,
`--card-unread-samples` for one session) trips: a `TRIP` row with `why`
`"card_unread"`, `reading` the count and `limit` the setting, the watched
tree killed and exit 3, as for the card. A reading back resets the count.
RAM and commit are still decided first, so a sample that trips both
writes `ram_guard` and exits 4. `tests/test_card_guard.py` holds it on the
stub host (three unread in a row trip; two, a reading, two do not, and
write a `card_unread` row each run; a limit of 1 trips on the first), with
no clock raced. No card or counter was read for it.

## Image metrics

`DEV#22`. A new file, `tools/image_compare.py`, and
`tests/test_image_compare.py`. It reads PNGs with Pillow and numpy. It
loads no model and computes no VAE round trip.

What it computes, using the definitions the card tables were made with:

- **Hash.** The gallery's own rule (`storage.content_hash`,
  `backend/storage.py:474`): the first 16 hex digits of the SHA-256 of
  `image.tobytes()`, the decoded pixels in the mode the PNG opens in. The
  tool computes it itself; a test imports `storage.content_hash` and holds
  the two equal on the same image.
- **Difference of A against B** (`card4-quant/compare.py`), both converted
  to RGB and promoted to int16. Mean absolute difference in 8-bit levels
  over every channel; the largest channel difference; PSNR as
  `10 * log10(255**2 / MSE)` over every channel, written as `null` with
  `"identical": true` when MSE is 0, because strict JSON has no Infinity;
  in JSON rounded as `compare.py` did (mean to 3 places, PSNR and the
  percentage to 2); the percentage of pixels whose largest channel
  difference is over the
  threshold (`--threshold`, default 16; the Nunchaku rows in verified.md's
  Behaviour section used 8); and the count of pixels that differ at all.
  Different sizes are refused, with exit 2 and both sizes named. The images
  are never resized.
- **Outside the mask** (`--mask`, any `L` PNG): the largest difference
  where the mask is 0. This is "Largest difference outside the mask 0" in
  the Inpaint tables, the check that `refine.composite` kept the source.
- **The seam step** (`--seam`, from `seam_insert.py`; verified.md,
  "The seam by number"). With the mask's 0 against above-0 boundary taken as
  horizontal and vertical neighbour pairs, the mean absolute difference
  across those pairs in each given picture (source, round trip if supplied,
  result), over every RGB channel. Where the mask has soft pixels
  (0 < level < 255), the mean of the summed horizontal and vertical
  gradients inside that region. The round trip picture is an input (the
  source composited over its own VAE round trip by `refine.composite`), made
  by the card session's own pipeline, never by this tool.

Output: one JSON object on stdout. `--row` also prints the line in
verified.md's column order: "mean / PSNR / % off by more than 16" (as in the
quantization tables) or "source / round trip / result" (as in the seam
tables), with the same rounding (mean to one place, PSNR to one place,
percentage to one place).

Constraints: stdlib, Pillow and numpy only (Pillow is declared; numpy comes
through `accelerate`, as above, and this item adds no dependency); the tool
imports nothing from `backend`, and only its test imports
`storage.content_hash`; the Pyright gate and ruff cover it. Its
numbers describe pixels already made, and the pictures themselves remain
the card's to produce.

Non-goals: strips, crops or diff images (each session's own `strips.py`
keeps those); perceptual metrics (SSIM, LPIPS) that no verified.md table
reports; judging by eye, which the tables keep as their own column.

### Image metrics stages and completion criteria

1. **CPU (done 2026-10-30), the only stage.** `tests/test_image_compare.py` on small
   synthetic PNGs made in the test, each with a known answer: identical
   images (MSE 0, PSNR `null`, 0 differing pixels); a single pixel off by 17
   on one channel (1 pixel over 16, max 17, the expected mean and PSNR to
   the digit); a uniform offset of 8 (0% over 16, 100% over 7); a mask whose
   outside region is untouched (max 0) and one with a planted change outside
   it (the change's size); a hard vertical mask edge with a known step on
   each side; a two-level soft band with a known gradient; a size mismatch
   refused with exit 2; and the hash equal to `storage.content_hash`. The
   CLI's `--row` output matches the expected string. ruff, the Pyright gate
   and the full pytest suite pass.

**Built (2026-10-30).** `tools/image_compare.py` takes one PNG (its hash),
two (A against B, with `--threshold` and `--mask`), or `--seam --mask m.png
source [round trip] result`; every picture's path, hash and size head the
JSON. The seam step and soft gradient are `seam_insert.py`'s measure as
that script wrote it: the gradient's forward differences count 0 past the
last column or row, and both are rounded to 2 places in JSON. `--row`
rounds the measured value to one place, never the JSON's rounded one (a
review fix: 0.0488% is 0.05 in JSON but 0.0 in the row, not 0.1). An image
and `--mask` without a second image is refused too. Choices the
plan left open, taken while Matt was away (2026-10-29), revisit if wanted:
a mask that is not mode `L`, a seam mask with no 0 / above-0 edge, three
images without `--seam` and a negative threshold are refused with exit 2
like a size mismatch; `--row` writes an identical pair's PSNR as
`identical`, a seam row without a round trip as `-` in that place, and the
gradient column as `-` when the mask has no soft pixels, joined to the step
by ` | `. `tests/test_image_compare.py` holds every case above; the pictures
and their numbers in verified.md remain the card's.

## CARD#10 step packs

`DEV#23`. `CARD#10` asks for fixes checked only on the fake to be checked on
the real server: 9377de1 (a fitted batch's columns from its first frame),
7733f81 (the strip fitted as if the page's scrollbar were there), 6a7570d
(a budget's seeds counted by the job's stop rule), ad01fdc (a sweep's
Resume held in words once its source is gone), 2a4d2cc (time left from the
first step's end), 2b89ed5 (a pass whose source went while it ran says so),
and 1744dc3 (R50) with a typed Ctrl-C, plus Ctrl-C to an idle server with
a model loaded (TODO.md's `CARD#10` row; in verified.md's Behaviour, the
first six are in the paragraph after the table and the two Ctrl-C checks
in the R50 row).
`git log` confirms all seven commits exist on 2026-10-05 and 2026-10-12.
The 2026-10-25 session that was to check them wrote `flux-krea/steps.py`.
Its packs drove the page through `startJob('/api/generate', collect())` and
`els.prompt.value`, and its one load stopped at the RAM guard. 6a7570d had
no pack there; it was first seen on a Kohaku XL Zeta grid.

The design: `tools/card/card10-<sha>.json`, one `tools/browser.mjs` step file
per fix commit, plus `tools/card/card10-ctrl-c-idle.json` for the idle
Ctrl-C, which has no fix commit. The rules for every pack:

- **Real input only.** The step under check is a `click`, `key` or `type`
  step on the element a person would use. The prompt goes in by `type`, so
  the page's own input path, Undo and Check run. No pack assigns `.value` to
  the prompt or negative (`AGENTS.md`: whole-box writes go through
  `writeBox`/`rewriteBox`/`replaceBoxText`, and a pack must not go round them). No pack
  calls `startJob` or a handler directly for the action under check. `eval`
  steps only observe (a MutationObserver, `getBoundingClientRect`, timings).
- **No `/api/fake/*` step** and no state planted that only a fake has.
  Expectations come from the smoke that covers each commit today
  (`docs/rules.md`, Tests). As `steps.py` did, the fake knobs are taken out
  and the waits lengthened for real steps. No single awaited expression runs
  past four minutes, since `browser.mjs` gives up on a call after
  `ANSWER_MS` (300,000 ms, `tools/browser.mjs:376`).
- **Wait for the adapter listing.** Before pressing Load or Generate, a
  pack waits until the page's first `GET /api/loras` has answered. It reads
  `performance.getEntriesByName` and sends no request of its own, because
  the `#lora` select's markup already carries `none`
  (`web/index.html:738`) and so shows nothing about the listing. 02cd1d9
  fixed the race this guarded against; the wait keeps each check about its
  own commit.
- **Each pack names its commit.** The first step is a `log` naming the sha
  and its subject, and the filename carries the same sha. A pack's own
  prerequisites are `{"expect": ..., "fatal": true}`, and async setup
  returns true explicitly (AGENTS.md, Tests).
- **Observe the page's rules, never work around them.** A held button is
  checked as `aria-disabled` with its reason as the title (ad01fdc's hold is
  now on the finish line's Finish run, `.finish-run-tool`,
  `web/runs.js:76`). Layout checks assert that nothing moved under the
  pointer and the page did not scroll by itself.
- Packs are not smokes. They are not in the port table, `run_smoke.py`
  never picks them up, and CI's browser job never runs them.
- The two Ctrl-C packs leave the signal to a person. 1744dc3's earlier
  check sent `CTRL_C_EVENT` from a helper process, which is "not a key
  press" (verified.md, the R50 row). `CARD#10` asks for a typed one, so the
  pack logs when to press it and waits for the page's "Server stopped"
  words.

Non-goals: a driver that starts the real server (Matt starts it, with
`--defaults` and a scratch gallery, as the 2026-10-05 rows did); any change
to `web/` or `backend/`; the critic's clock and size labels (b396e2e,
5ec8adb), which belong to the Prompt analyser section and to `CARD#11`.

### Step pack stages and completion criteria

1. **CPU and fake (built 2026-10-30 but for the typed Ctrl-C on the fake,
   which is Matt's).** The eight packs and `tests/test_card_packs.py`.
   The test checks that every step is valid JSON of a kind `browser.mjs`
   accepts; that no step mentions `/api/fake/`; that no step assigns
   `els.prompt.value` or `els.negative.value` or calls `startJob(`; that
   each commit pack's filename sha appears in verified.md's Behaviour
   section (the paragraph listing the `CARD#10` commits, or the R50 row for
   1744dc3; text only, since CI's test checkouts are shallow) and matches
   its first `log`; and that each pack waits for
   the adapter listing before its first Load or Generate press. A dry run on
   a fresh FLUX.1-Krea-dev fake (`--data` scratch, `--step 0.3`) passes
   every page pack. On the fake, the two Ctrl-C packs are run with Ctrl-C
   typed by hand in the fake's console, and the result is recorded.
   `docs/rules.md` gains a `tools/card/` Layout entry saying the packs are
   not smokes. ruff, the Pyright gate and the full pytest suite pass.
2. **+card, with Matt's OK.** The real server with FLUX.1 Krea [dev] at fp8
   and model offload, which needs the RAM cost stated first (the fixed
   loader peaked at 27.9 GiB working set, 14.2 GiB free at the lowest from
   44.7), and Matt at the console for the typed Ctrl-C. The packs run one at
   a time with `DEV#20`'s guard watching the server. Each result becomes a
   verified.md Behaviour row, and `CARD#10` closes in TODO.md only when every
   pack has run.

**Built (2026-10-30), stage 1 but for the typed Ctrl-C.** `tools/card/`
holds `card10-9377de1.json`, `card10-7733f81.json`, `card10-6a7570d.json`,
`card10-ad01fdc.json`, `card10-2a4d2cc.json`, `card10-2b89ed5.json`,
`card10-1744dc3.json` and `card10-ctrl-c-idle.json`. Each navigates to
`http://127.0.0.1:7860/` (the real server's port; `AURADREAM_PORT_OFFSET`
moves it to a fake's), waits for the page's first `GET /api/loras` in its
resource timing, then presses Escape over the Models picker, which a
server with nothing loaded opens over the panel at its first status, and
checks it closed. Fields are filled by a click, Ctrl+A and `type` (W and H
sit inside the closed Custom dimensions disclosure, opened by its summary
first); the page is wheeled to a gallery card or the Bin button, never
scrolled from an eval. Sizes were chosen for the card: the two batch packs
render four 512x512 images (1 MP at once), 2a4d2cc one 1024x1024 image at
12 steps, the budget pack 768x768 (a size no other pack renders, so the
batch packs' per-image times, which the gallery's per-step median would
otherwise take, do not price it: on the first dry run they did, and the
preview said 12 of 12), the refine packs a 768x768 source at 28 steps. What
each checks, from the smoke that covers its commit today:

- 9377de1 (smoke-results): at 2560x1305, a MutationObserver on the strip
  holds the live preview's left edge at the first image's slot and every
  tile's left edge still while the four land, `--cols` 4, the page
  unscrolled and Generate unmoved under the pointer.
- 7733f81 (smoke-results): the same batch; `--tile-h` and the gallery head
  unchanged through the end and 1 s after it. It logs `barsSeen`: the case
  the fix is for (a scrollbar that comes with the progress block and goes
  at the end) needs a gallery too short to scroll the page, so the pack
  says to run it first on a fresh scratch gallery. The pack does not force
  the scrollbar as the smoke does (`overflowY` on `html` and `fitViewer()`),
  since its evals only observe.
- 6a7570d (smoke-grid, `tests/test_api_grid.py`): three images priced, then
  Steps 8,16 x 12 seeds with a 1 min budget; a fatal check that the
  preview says some seeds and not all (else the budget does not fit the
  machine's pace), then Run grid, and the seeds the stop left complete
  (kept cells over two) equal to the number the preview said.
- ad01fdc (smoke-refine): a six-cell sweep at 1x, strengths 0.3-0.8; once
  the first cell has landed the source is deleted from its gallery card's
  ⌦ and the recycle bin emptied through its dialog; the stop names "no
  gallery image with id N", and Finish run (`.finish-run-tool`, Resume's
  place since) pressed twice holds as aria-disabled with that reason as its
  title, keeps the focus, and sends nothing (`submissions` unchanged).
- 2a4d2cc (smoke-jobs): the step counter and the cell line observed; no
  "left" at step 1, some estimate later, and every estimate at most twice
  the time the job then had left plus 5 s. On the fake every step takes the
  same time, so the check only bites on the card, whose first step carries
  the transformer's move onto the card.
- 2b89ed5 (smoke-refine): a pass at strength 0.9; once its first step has
  landed the source is deleted and the bin emptied (fatal: the pass still
  running); the pass lands with `refine_missing`, "(deleted)" in its title
  and the note under the strip, and its Reuse saying the source has been
  deleted.
- 1744dc3 and the idle Ctrl-C: fatal unless nothing is loaded (1744dc3) or
  the image just made left the model loaded and the server idle; then a
  `>>> NOW` line asks for one typed Ctrl-C, and two eval waits of 4 min
  each (an eval does not fail) give the person 8 min before the checks:
  the title "Server stopped …" and "Stopped: the server is shutting down —
  nothing had finished." (1744dc3), the topbar's "server not answering —
  retrying" and its dot `error` (both). The pack logs every change of the
  title, the line under Generate and the topbar, timed from the ask; the
  server's side (rc, no second Ctrl-C, no Traceback) is read off its
  console. Both ad01fdc and 2b89ed5 empty the whole recycle bin, so they
  say to run on a scratch gallery.

`tests/test_card_packs.py` holds the eight names (none a smoke, none in
AGENTS.md), every step one of the kinds `browser.mjs`'s step loop
dispatches on with only that kind's modifiers, no `/api/fake/`, no
`els.prompt.value =`/`els.negative.value =`, no `startJob(` and no
`fetch(`/`api(` of a pack's own, no `.click(`, `.focus(`, scroll call or
`.value =` from an eval, no wait or `__until` past 4 min, each
commit's sha in verified.md's Behaviour paragraph (or R50's row) and at the
head of its first `log`, a fatal wait on the listing before the first press
of Load, Generate, Run grid, Refine or Switch & load, and in the Ctrl-C
packs one ask and no Ctrl-C sent. `docs/rules.md`'s Layout has the
`tools/card/` entry.

Dry run, 2026-10-30, on a fresh FLUX.1-Krea-dev fake (`--step 0.3`, a new
scratch `--data`, port 8138 with `AURADREAM_PORT_OFFSET=278`), at the
packs' own viewports, in the order 7733f81, 9377de1, 2a4d2cc, 6a7570d,
2b89ed5, ad01fdc: every page pack exited 0. 9377de1: the preview at the
first slot and four tiles still, `--cols` 4. 7733f81: `--tile-h` 491px and
the head at 727.5 throughout, but the scrollbar (15 px) came with the first
frame and stayed, the four new cards scrolling the page, so the
comes-and-goes case was not exercised on the fake. 2a4d2cc: no estimate
until step 4 (the fake's elapsed reaches 1 s there), then "~2s left" with
2.7 s left down to "~0s" with 0.4 s. 6a7570d: "about 88s … budget 1 min ≈ 8
of 12 seeds complete", stopped after 16/24 cells, 8 seeds. 2b89ed5: the
pass #32 "refine of #31 (deleted)". ad01fdc: "Stopped at cell 5 of 6 after
3 cells failed; 2 images kept", Finish run held on "no gallery image with
id N" at both presses. The two Ctrl-C packs were run up to their `>>> NOW`
line only, without a signal: the idle one after its image, the dot ready;
1744dc3 on that fake after `POST /api/model/unload` and `POST
/api/fake/load {"seconds": 30}` sent by hand (the fake starts loaded and
loads at once, and a pack sends neither), "loading weights" and the dot
`loading` 5 s on. A review the same day had 2b89ed5 check the card it deletes is
the source's and 2a4d2cc that the size is 1024x1024, and ran the six page
packs again on a fresh fake (port 8150): all exited 0 with the same
readings (8 of 12 seeds, 16/24 cells; #32 "refine of #31 (deleted)"; cell
5 of 6 after 3 failed).

**Hardened after review (2026-10-31): the bin packs refuse a real
gallery.** ad01fdc and 2b89ed5 navigate to 7860 and empty the whole
recycle bin; nothing checked what it held, so a run against Matt's
gallery, or a dry run without `AURADREAM_PORT_OFFSET` while the real
server is up, would have deleted every binned image for good. Each now
stops (fatal) at two points, reading only the page's own state. Before
its first press, unless the page opened on a scratch gallery: once the
gallery's first page and the bin summary have answered, no image under
the checkpoint (`total` 0, no card, no search, Starred or All models), the
bin empty (`binCount` 0), and no row on any checkpoint past #2
(`galleryReadHead`, the page's newest id over every row: null on a fresh
real scratch gallery, 2 on a fresh fake, which plants two under another
checkpoint). And between opening the bin and pressing Empty bin, unless
the bin holds the source alone (`binCount` 1, one record and one card,
each the source's id, no further page). So each bin pack runs on its own
fresh scratch gallery: the second on the same fake now stops. The test
finds every pack that presses `#btn-empty-bin` (these two), and requires a
fatal check of the bin's contents between the last `#btn-bin` press and
each Empty bin press, and a fatal scratch check before the first press
that could load; both failed on the packs as they were and pass now.
Dry run (port 8150, `AURADREAM_PORT_OFFSET=290`, `--step 0.3`, a fresh
scratch `--data` per pack): ad01fdc exited 0 (opened on total 0, bin 0,
newest #2; the bin held `[3]` alone; "Stopped at cell 5 of 6 after 3 cells
failed; 2 images kept", Finish run held); 2b89ed5 on that same fake
afterwards exited 1 at its scratch check (total 2, newest #5) with nothing
typed or pressed; on a second fresh fake it exited 0 (bin `[3]` alone; #4
"refine of #3 (deleted)").

**Left, Matt's:** the two Ctrl-C packs on the fake with Ctrl-C typed in the
fake's console, and their result recorded here. For 1744dc3 the fake must
first be unloaded and its load slowed by hand, as above (for example
`curl -X POST -H "Content-Type: application/json" -d "{\"seconds\": 120}"
http://127.0.0.1:7861/api/fake/load` and `curl -X POST
http://127.0.0.1:7861/api/model/unload` against a fake on 7861, then
`AURADREAM_PORT_OFFSET=1 node tools/browser.mjs
tools/card/card10-1744dc3.json`). Stage 2 is unchanged.
