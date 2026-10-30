# Agent expert walkthrough — 2026-10-30

This is an agent review of a disposable fake gallery for the refreshed kit
(`DEV#21`, fixture version 2). It is not a human trial, a timing comparison,
GPU qualification or a redesign decision. Matt's predictions, timings, quotes
and outcomes stay blank in the [human record](observation-template.md). The
[2026-10-18 record](expert-baseline-2026-10-18.md) is the version 1 history
and is unchanged.

## Build and environment

App sources were commit `ce88c55` (main, merge of pull request 21). The
working tree held only the uncommitted kit changes of this refresh
(`tools/trial_fixture.py`, `tools/trial_walkthrough.py`, the trial docs,
`docs/rules.md`, `tests/test_trial_kit.py`); `git diff ce88c55 -- web
backend` was empty, so the served page and fake were the commit's. The
browser reported Edge/HeadlessChrome 156.0.0.0 on Windows NT 10.0, 1400×900
CSS pixels, DPR 1 (`AURADREAM_VIEWPORT=1400x900`, `AURADREAM_DPR=1`, no port
offset), default automation zoom, its own temporary profile per run; Node
24.14.0. Monitor resolution and Windows scaling were not measured: this is a
headless desktop baseline, not Matt's workstation. No weights, GPU, Hub or
live gallery were touched.

The fixtures used ports 8159 (A) and 8160 (B), away from the smoke table and
from the kit's default 8312, because other agents ran smokes on this machine
at the same time. Each was a new owned scratch directory under the session's
temp folder (`trial-v2-a9`, `trial-v2-b1`), with 769,065,365,504 and
769,212,256,256 bytes free before setup and about 2.6 MB each after the
whole run. Each fake server was stopped afterwards and its port checked
closed.

Commands, per variant:

```powershell
uv run --no-sync python tools/trial_fixture.py --data <new scratch> --port 8159 --variant A
uv run --no-sync python tools/trial_walkthrough.py --manifest <scratch>/fixture.json --output <new steps.json>
uv run --no-sync python tools/fake_server.py --port 8159 --model black-forest-labs/FLUX.1-Krea-dev --data <scratch>
node tools/browser.mjs <steps.json>     # AURADREAM_VIEWPORT=1400x900, AURADREAM_DPR=1
```

## Fixture verification

Both variants prepared without error. Each manifest recorded 69 live + 1
binned = 70 records: the 60-cell complete grid, 2 of 12 interrupted images,
source #65 and child #66, rejects #67 (binned) and #68, foreign #1/#2, and
the Then refine base #69 with its pass #70 (384×384, strength 0.5, a run of
its own, the same job). One setup (`Lighthouse setup A` / `Observatory setup
B`), one saved search (`Lighthouse grid A` / `Observatory grid B`, `run:` of
the complete grid) and one snippet (`Harbor light A` / `Terrace light B`)
were listed by their routes. The three refusals were the server's own 422s:

| Request | Answer |
|---|---|
| Re-render of a foreign image | "1 of the selected images was made on krea/Krea-2-Turbo; the server is running black-forest-labs/FLUX.1-Krea-dev, whose steps, CFG and mu differ. Unpick #2." |
| Refine from the binned reject | "source image 67 is in the recycle bin; restore it to use it" |
| More seeds of a foreign image | "no selected batch can have more seeds (1 foreign)" |

## Agent observations, not participant outcomes

Both runs passed **206 of 206** expectations and exited 0. The counts below
are trusted automation events (A and B were identical), not Matt's presses;
inserted text and wheel movement are not comparable to physical effort, and
scroll-to-target steps do not establish human reachability.

| Task | Real inputs and observed outcome | Detours, context and focus |
|---|---|---|
| 1 Make / Reset | Prompt, True CFG 1.5, negative, 7:9, 1 MP, seed, Batch 2, Repeat 2, Steps 1: four portrait images. Reset cleared Prompt and Seed and focused Prompt; Undo restored the panel exactly; a second Reset took the new seed. 15 clicks, 16 keys. | FLUX's True CFG before the negative, and an upward wheel to reach Reset in the tall panel, as on 2026-10-18. |
| 2 Busy queue / cancel / return | A1 running, A2 waiting, child inspected, A3 queued; A2 cancelled by its row's named × button, then A1 by Cancel: both cancelled with no image. Reload kept the draft and watched A3, which made one image. 15 clicks, 18 keys. | One press on the closed Custom dimensions summary to reach W/H (compact setup, `UI#19`). After the reload the disclosure was open again with W 256. The reload left no control focused. |
| 3 Inspect / sheet | Child #66 opened, compared with source #65, unstarred and restarred; `?` opened keyboard help and Escape closed it; `/` focused search for the complete run; sheet, Fill width, back to the child. 10 clicks, 14 keys. | The draft persisted throughout. Closing the lightbox returned focus to the child's image. |
| 4 Reuse / picked rerun | In the keeper's lightbox, M opened the More seeds preview with nothing queued, and Escape returned to the lightbox. Reuse replaced the draft; its Undo restored it. W/H 512, picks keeper then child through `/` searches, Starred on: Re-render made exactly two 512×512 images; picks and draft kept. 12 clicks, 19 keys. | The size disclosure was already open (from task 2), so no second press. |
| 5 Archive / restore | Whole gallery, password, 77 records (70 + 7 made above) backed up, typed count, Bundle & clear, Download saved bundle, unlock and import: 77 imported, 1 binned, the foreign rows back. The setup, saved search and snippet were still listed by their ids. 11 clicks, 12 keys. | Clear dropped the picks; the draft persisted. |
| 6 Restore a setup / Undo (optional) | Examples, Saved setups…, the setup's row, Restore: prompt, 512×768 and its seed in place, nothing queued. The prompt note's Undo restored the whole draft exactly. 6 clicks, 2 keys. | Restore waits for its own validation; no press was held. |
| 7 Saved search (optional) | Footer Saved searches, the search's row, Apply to gallery: the box read the `run:` query, Starred went off, 48 cards of the 60-image run shown (one page). 3 clicks. | Focus returned to the footer opener. |
| 8 Then refine (optional) | Steps 8, Then refine on at 1.5× and strength 0.5, Generate + refine: one 512×512 base and its 768×768 pass, a run each, one job. Searching the prompt's words showed exactly the two; the pass's Compare with source showed the base. 11 clicks, 10 keys. | Three detours below. |

### Detours and findings

- **The panel's lower rows below the window (task 8).** After tasks 1-7, at
  1400×900 the panel was 1139 px tall with its sticky top at 68 px; Then
  refine's opener sat at 1026 px and a programmatic scroll could not raise
  it. That is the panel's design (`web/panel.js`: it moves with the user's
  own scrolling, never the page's); one real downward wheel brought it into
  view. A participant at a short window meets the same scroll.
- **The pass strength (task 8).** The Then refine dialog started at
  strength 0.1. With Steps 2 the server refused Generate + refine in its own
  words, "strength 0.1 of 2 steps leaves no step; set the pass's Steps or
  raise the strength", and nothing queued. The driver now names Steps 8 and
  strength 0.5. A clear refusal, recorded, not a defect.
- **Select mode outlives clear and import (task 8).** Task 4's Select mode
  was still on after the archive (its picks were gone), so a card click
  would pick the pass rather than open it; one press on Select left it.
- **The version 1 driver passed while missing.** Before the repair, the
  2026-10-18 driver also exited 0 (54 of 54) on this tree, but its task 2
  W/H fills missed the hidden fields: the survivor job rendered at
  1024×1024 instead of 256×256, and no expectation looked. The repaired
  driver checks each fill's focus and each press's target, and task 4's
  rerender size.

### Diagnostic history

Six earlier full runs and one partial run, each on a fresh A fixture, were
diagnostic and are not completion evidence: they found the reload dropping the driver's request
log, an empty search the driver could not type, the three task 8 detours
above, and once (before the target check existed) a Back up wait in task 5
that timed out; that one did not recur in the six later runs and its cause
was not established. Only the final A and B runs are the baseline.

## Limits

These are expert fixture observations on synthetic pixels and fake timing,
not usability rates. Matt's session (`DEV#1`), his display scaling and
input, screen reader and touch work, GPU qualification and real speed
comparisons remain unverified. The live-server reload is not restart
recovery. No application source was changed for this record.
