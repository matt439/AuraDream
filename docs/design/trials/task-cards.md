# First short task set

Give the desired outcome, then observe without naming controls. Before any
Queue/Generate/clear, ask the participant to predict the target, count and
what remains. A/B are task variants, not interface names. Use A for one
interface and B for the other; alternate on repeats and record prior exposure.
Fake generation waits are separate from interaction time. The observer may
split this set into short sessions.

| Task | Variant A | Variant B |
|---|---|---|
| 1. Make, Reset, recover | Prepare “An observatory above clouds”, negative “blur”, portrait 7:9, 1 MP, seed 4101, Batch 2, Repeat 2. Predict the total and make it. Reset; inspect Seed; recover the draft; Reset again and enter 5101. | Use “A greenhouse beside a lake”, negative “low contrast”, seed 6101, then 7101; same aspect/tier/counts. |
| 2. Work while busy, cancel, return | Observer starts named long fake job A1 and waiting A2. Inspect the earlier child while work runs; prepare and queue A3. Identify the job being watched. Cancel A2 and then A1. Reopen the tab and find A3 and the saved results. | Use B1/B2/B3 and the alternate child. Keep queue lengths/runtime conditions the same; reverse which earlier image is inspected. |
| 3. Judge and orient | Find child, read prompt/settings, compare with source, star it. View the completed variable/Steps/CFG run as a sheet, enlarge its image space, then return to the child. | Same relationships and grid shape with alternate subject/seeds. |
| 4. Reuse and picked rerun | Keep “Unfinished harbor draft” in the composer. Inspect keeper; predict what taking its settings replaces; do it and recover the draft. Pick keeper then child across searches, leave Starred on, predict which images Re-render will make and at what panel size, then do it. | Keep “Unfinished mountain draft”; alternate keeper/child; identical two-image scope and panel size. |
| 5. Finish curation and restore | With search/current-model filter, Starred and picks active, back up the **whole** scratch gallery with password `study-only-A`. Archive-and-clear it, download the saved bundle, then import it. Predict scope before clear/import, and what the clear leaves behind: the gallery's images and bin versus the saved setup, saved search and snippet. Find the restored live images and bin, then check the `setup`, `saved_search` and `snippet` are still listed. | Same counts/bin mix with alternate subject, password `study-only-B`. |

Use `fixture.json` aliases, not guessed IDs. For current-app busy setup, the
observer can use the fake-only `/api/fake/step` control to make work slow
enough to inspect/cancel; record that setup separately. Use normal page
Generate/queue controls for the participant. Clear the fake delay afterward.
For an automated expert walkthrough the generated driver records its fake
setup separately; it does not model human discoverability or timing.

For the concept use Run A, keeper #101, child #107/source #101, picks #101
then #107; generation task values stay A/B as written. Select Gallery,
Starred and the current-model filter for the archive scope exercise. The
concept has no actual bin or archive files. Its simulated queue J12/J13 is a
separate exercise: identify watched work, cancel the requested row, inspect
the retained draft. Reopen/reconnect, real planner/execution, verified clear,
encrypted download/import and full Reset persistence are **not represented**.
Record their absence; do not award a success, failure or speed advantage.

Observe four-image prediction and separate Batch/Repeat; visible empty Seed
and useful Undo; correct cancellation target and live-server reconnect;
source identity, sheet orientation, draft/focus on return; explicit draft
replacement and recovery; two retained picks instead of visible filter scope;
and whole-gallery counts including another checkpoint and the bin. Record
wrong-scope attempts and refusals even if no request is submitted.

Do not require Pause, Gallery beside, a time budget, explicit Seed list,
sheet export, loader changes, painted edits, Then refine, saved setups,
saved searches, snippets or setup/snippet export and import in this first
set; a participant may use any of them as a route, recorded as such. These
are synthetic task values, not recommendations or preferred generation
settings.

## Optional second set

Offer these only after the first set, and only if Matt wants them in his
trial. Record which were offered. Matt rated saved searches and Then refine
**Sometimes** on November 3. Saved setups have no rating. Prompt variables
are **Often**, Steps/CFG sweeps **Sometimes**. Use those ratings to choose
tasks. Score each attempted task from the predictions and actions observed
in the session. The task 5 import renumbers gallery images, so these cards
use sidecar names and new images. [Reported use](../preferences.md#later-features-2026-11-03)

| Task | Variant A | Variant B |
|---|---|---|
| 6. Restore a setup, then Undo | Keep “Unfinished setup draft A” in the composer. Predict what restoring `setup` (“Lighthouse setup A”) replaces and whether it queues anything. Restore it, check prompt, size (512×768) and seed, then recover the draft. | “Unfinished setup draft B”; `setup` is “Observatory setup B”. |
| 7. Reopen a run through a saved search | Predict what applying `saved_search` (“Lighthouse grid A”) does to the search box, Starred, All models and picks. Apply it and find the completed grid's images. | `saved_search` is “Observatory grid B”. |
| 8. Then refine | Prompt “Trial then refine A”, Steps 8, Then refine at 1.5× and strength 0.5. Predict how many images and at what sizes, then Generate. Find the pass beside its base and compare the two. | “Trial then refine B”, same numbers. |
| 9. Vary a guided tag prompt | Start from Animagine's green-haired girl example. Keep the example and vary the light between moonlight and sunlight using one prompt variable. Cross those two choices with Steps 2 and 4 at seed 4701. Predict the resolved prompts and total, run the experiment and identify one output for each pairing. | Start from the green-haired boy example. Use sunlight and candlelight, Steps 2 and 4, seed 6701. |

For card 9 the observer prepares the Animagine fake with a blank draft, Then
refine off, one image per batch, one seed and other sweep fields empty. Ask
for the prediction before submitting. The observer checks four results,
each of the two prompts at both step counts, all sharing the fixed seed.
The concept lacks this execution path. Record only the steps it represents.

The fixture's `then_refine_base`/`then_refine_pass` pair is a made example;
show it only after task 8, if at all. The walkthrough also records the
detours these tasks met on the fake (the panel's lower rows, Select mode,
the pass strength); the participant's own route is what is observed.

## Follow-up comparisons for DEV#1

Compare the reviewed build `d072868f514e` with the revised build. Use a
fresh scratch fixture for each attempt. The observer prepares an unfinished
prompt and fixed seed. Alternate A/B values and layout order on
repeats. Keep mouse and keyboard routes available. Record the participant's
prediction before opening any preview or review that reveals the answer.

1. **What will a bulk action affect?** Pick `keeper` and `child` through
   separate searches, then search for `source`. Name the affected image IDs,
   remove just `keeper` and predict the remaining scope. Leave the search
   and draft in place. On the baseline use the existing count and selection
   controls. On the revised build allow Review picked. Record predicted and
   actual IDs before and after the removal, detours to another filter and
   active interaction time for both attempts.
2. **Can you return from judging to the intended work?** Inspect `child`,
   compare it with `source`, return to the unfinished draft and make two
   more seeds of the child. Ask which image will supply the settings before
   the participant confirms. Check the source and
   target IDs and the draft text, fixed seed and filter on return. Record
   any reconstruction, mistaken target or detour before the press. For the
   portrait viewing probe, prepare one portrait from task 1. In the gallery,
   change thumbnail size and switch between the whole image and its cropped
   thumbnail at the same sub-1800 CSS viewport in both builds. Record the route,
   presses and any need to reopen or relocate the image.
