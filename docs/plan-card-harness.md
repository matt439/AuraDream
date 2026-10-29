# Plan: a checked-in card-session harness

**Status:** `DEV#20` (card guard), `DEV#22` (image metrics) and `DEV#23`
(`CARD#10` step packs) opened 2026-10-29 from that day's TODO audit, which
Matt approved in full. Nothing is built. Each item's first stage is CPU,
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
changes the defaults there and nowhere else. Neither exists yet.

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

1. **CPU (OPEN).** `tools/card_guard.py` and `tests/test_card_guard.py`,
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

1. **CPU (OPEN), the only stage.** `tests/test_image_compare.py` on small
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

1. **CPU and fake (OPEN).** The eight packs and `tests/test_card_packs.py`.
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
