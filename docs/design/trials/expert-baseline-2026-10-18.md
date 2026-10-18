# Agent expert walkthrough — 2026-10-18

This is an agent review of a disposable fake gallery and disconnected
concept. It is not a human trial, a timing comparison, GPU qualification or
a redesign decision. Matt's predictions, timings, quotes and outcomes are
blank in the [human record](observation-template.md).

Build, repeatable commands, actual inputs, observations and limitations are
recorded below after verification. Fixture preparation uses real application
APIs; browser task actions use real input. Setup/instrumentation is excluded
from participant observations.

## Fixture verification

`tools/trial_fixture.py --variant A --port 8312` passed with new scratch
`local-diffusion-trial-A-d715c5177e904aa6bd0c8e7106fb566b` under the system temp
directory. At setup, C: had 1,197,080,121,344 bytes free. The manifest recorded
67 live + 1 binned records, 60 completed cells, and 2 of 12 images before
cooperative cancellation. Keeper #3, source #65, child #66, binned reject #67,
live reject #68; foreign rows #1/#2. No weights or live data were accessed.
The scratch directory was 692,022 bytes after setup. Its positively
acknowledged Windows launcher/runtime PIDs were 106368/106432; both the
runtime shutdown and closed port were checked. An earlier immediate port
check reported the listener while it was closing; the bounded five-second
shutdown check passed on this new fixture. No unrelated process was stopped.

Variant B also passed at
`local-diffusion-trial-B-b05a896afcfc4a12899efd4501dab684`, with the same counts
and alternate prompts/seeds. A rerun against that existing path was refused
before mutation. Foreign Re-render returned 422 with the checkpoint mismatch
and `Unpick #2`; binned-source Refine returned 422 with the instruction to
restore #67. These are actual server refusals, not simulated planner answers.

The existing disconnected concept `prototype/check.json` passed all 54
expectations at port 8311 (`LOCAL_DIFFUSION_PORT_OFFSET=400`). The static
server was stopped afterward. Its 6,435-byte log is
`%TEMP%/local-diffusion-dev3-concept-20261018.log`. This covers synthetic
Reset/Undo, Batch/Repeat, queue cancellation, source comparison, sheet Fill,
picked scope and the archive-route explanation; it does not execute archives
or recover a restarted server.

## Current-app build and environment

The five-task baseline used commit
`17c7ef031762ba17f1189277991fb0e1d01f737f` on main, including the committed
Finish run page integration. `git archive 17c7ef0` supplied an isolated source
snapshot, so the shared checkout's concurrent dirty edits were not served.
The snapshot had no working-tree edits; the new DEV#3 driver was external
test tooling. Snapshot ZIP: 3,880,916 bytes. Source location:
`%TEMP%/local-diffusion-dev3-source-ac23ceb02a6c40a58819d43e30f60a49`.

The fake server ran that snapshot on port 8312 against the prepared A fixture.
The browser reported Edge/HeadlessChrome 155.0.0.0 on Windows NT 10.0,
1400×900 CSS pixels, DPR 1, default automation zoom. Actual monitor resolution
and Windows display scaling were not measured. This is a headless desktop
baseline, not Matt's normal workstation/scaling session. Each browser run had
its own temporary profile. No application sources were changed for the trial.

Commands: generate `walkthrough.json` with `tools/trial_walkthrough.py
--manifest <scratch>/fixture.json --output <new steps.json>`, then run
`node tools/browser.mjs <steps.json>` with no port offset, viewport 1400x900
and DPR 1. The [kit instructions](README.md) give the complete launch/reset
sequence. The committed snapshot's `tools/fake_server.py` was used rather
than the concurrently edited shared source tree.

## Agent observations, not participant outcomes

| Task | Real inputs and observed outcome | Context, focus and detours |
|---|---|---|
| Make / Reset | Entered prompt, True CFG 1.5, negative, 7:9/1 MP, seed 4101, Batch 2, Repeat 2 and Steps 1; Generate made four portrait records. Reset cleared Prompt/Seed and focused Prompt; Undo restored the complete recorded panel; a second Reset allowed seed 5101. | FLUX required a separate True CFG adjustment before the negative could be typed; the WAI concept does not. Upward wheel navigation was needed to reach Reset in the tall sticky panel after Generate. The final focus was Seed. |
| Busy queue / cancel / return | Queued A1, A2 and A3 through Generate. Inspected child #66 while A1 ran. Cancelled A2 by its row, then A1; both terminal states were cancelled with zero saved images. Reloaded while A3 still ran; the page watched the same job and retained “Trial survivor A3”. A3 then completed one image. | The existing page automatically followed the surviving job; the concept's watched-job selector represents a different interaction. Earlier results were inspectable without replacing the composer. Reload focus was Body; this is live-server tab reconnect, not server restart recovery. |
| Inspection / sheet | Opened #66, read its prompt, compared with source #65, explicitly unstarred/re-starred it, displayed the completed run as a sheet, pressed Fill width and returned to #66. | The unrelated draft persisted. Closing inspection returned focus to the child's image. Sheet orientation came from real row/settings data; no export was needed. |
| Reuse / picked rerun | Entered “Unfinished harbor draft”; reused keeper #3 and observed replacement; Undo restored the draft. Set the panel to 512×512. Picked #3 then #66 across searches; with Starred on and #3 hidden, Re-render made exactly two images at the new size. | Picks [3,66] and the unrelated draft survived the rerender. Focus remained Re-render. Today's Re-render label was its confirmation; there was no invented preview/Queue dialog. |
| Whole-gallery archive | With current-model search, Starred and two picks active, used password `study-only-A` to back up the whole gallery, then typed 75 and bundled/cleared it. Pressed Download saved bundle, unlocked and imported the saved protected bundle. Actual API counts confirmed 74 live + 1 bin, including the other checkpoint. | Backup kept all 75; verified clear yielded zero records; import restored 75 including the bin. Clear removed the old picks. The unrelated composer draft persisted. The pressed download route is exercised; no claim about a human file-picker choice or disk destination is made. |

The driver logged trusted click/key **events**, not human presses: task 1
15/16, task 2 14/18 before reload, task 3 11/11, task 4 14/15, task 5 11/12.
Modifier events are counted separately; inserted text and wheel movement are
not comparable to physical typing effort. Helper scroll-to-target steps
also mean these numbers do not measure Matt's navigation cost. No personal
completion times, predictions or feedback were supplied.

## Failures, correction and retained evidence

The first full walkthrough traversed all five workflows with **53 of 54
expectations passing**. Its immediate Reset assertion failed. A focused
reproduction showed the synthetic click had not reached Reset: Prompt/Seed
were unchanged, `resetNote` was false, and the focus-event log contained no
second Reset focus. The first run's following Undo equality was therefore
not accepted as Undo evidence. A simple Reset probe passed.

The driver now uses a real upward wheel before Reset. The corrected
generation/Reset/Undo task passed **8 of 8** checks, explicitly observing empty
Seed/Prompt, Prompt focus, full-panel Undo and a newly entered seed. No app
focus defect was established or fixed. A diagnostic probe that pressed Reset
twice predictably failed its old-draft Undo assertion; that probe is retained
as diagnostic output and excluded from completion evidence. The other four task paths passed in the first full run. A final complete
corrected run then verified all five task boundaries together, as below.

Small text evidence remains under the A fixture directory:

- `walkthrough.log`: all five tasks, the original single failed assertion,
  browser/build context and actual archive counts.
- `generation-reset-probe.log`: exact unchanged fields and missing Reset
  focus on the missed-click path.
- `reset-probe.log`: normal Reset fields/focus.
- `generation-reset-wheel-probe.log`: diagnostic double-Reset probe, excluded
  from pass evidence.
- `generation-reset-final.log`: corrected first task, 8/8 passing.

The owned snapshot fake server was stopped after the checks. Its fixture and
text evidence totalled 2,740,906 bytes after the diagnostic probes; C: still
had 1,182,189,457,408 bytes free. These are expert fixture observations, not
usability success rates. The first
human task set, actual scaling, screen reader/touch work, GPU qualification
and real-user speed comparisons remain unverified. The concept's simulated
Finish and this baseline's live-server reload do not establish restart
durability; GEN#11's own recovery verification remains separate evidence.

## Final complete corrected-driver verification

The corrected reusable driver passed all **54 of 54 expectations** in one
complete five-task run against the same committed application snapshot
17c7ef0 (branch main), on an owned fresh A fixture at port 8812. Browser
viewport/DPR were explicitly 1400x900/1. Generation/Reset/Undo, live-server
reconnect, source/sheet inspection, hidden-picked rerender and the actual
75-record protected backup/clear/download/import all passed together.
The prior 53/54 and focused 8/8 results remain diagnostic history.

Evidence: %TEMP%/local-diffusion-dev3-final-w8vduqpl/walkthrough.log and
result.json. The complete retained source ZIP/snapshot, fixture and logs
occupied 20,289,460 bytes. App sources came from git archive17c7ef0; only the
external trial tools were copied into that owned snapshot. No production
source was modified for the check. An initial port8312 attempt refused
before fixture mutation because an abandoned owned trial worker remained;
its exact runtime/parent/source/data identity was checked before cleanup.
The final8812 worker and listener were also verified stopped. No unrelated
service was modified. This is technical preparation, not a human trial.
