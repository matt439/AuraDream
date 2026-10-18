# Options and validation

Status: design proposal, 2026-10-18. No human task trial has been
run for this work. The comparisons, methods and targets below are proposals.
They are not usability measurements or approval to change the application.

Scope: Matt's personal workstation. Existing operating modes describe one
person's changing tasks; they do not stand in for interviews with other users.
Matt explicitly prioritized doing common tasks quickly. Interaction time,
detours and retained context lead the evaluation; model runtime is separate.
Start with the [design index](README.md), [research and journeys](research.md),
[workbench proposal](workbench.md) and [requirements](requirements.md).

## Three directions

The comparison concerns the whole application's organization. Colour, type
and component styling cannot resolve its context and scope problems alone.
Assess each option against the make-judge-extend loop, overnight continuity,
curation, action-scope clarity, attention cost and implementation disruption.

| Direction | Benefits for Matt | Costs and uncertainties |
|---|---|---|
| Cohesive workbench with task lenses | Keeps the composer, results and queue together; current results, runs and library can receive attention without leaving the workstation; fits existing surfaces | Requires clear scope and view state; an overfilled inspector could recreate the current density |
| Hard task workspaces: Make, Explore, Review | Gives planning and curation generous space; reduces simultaneous controls | Making and judging alternate after each batch; repeated workspace changes could add friction and obscure the draft or current job |
| Project or run notebook | Gives overnight work a durable home for definition, outcomes and images; makes run-wide actions natural | A project hierarchy is not evidenced; run durability depends on retained cells; single-image iteration could become a trail of notebooks |

**Recommend the cohesive workbench as the first hypothesis to prototype.**
Borrow the task workspaces' focus and the notebook's durable run context,
while preserving direct movement between making and judging an image.
The recommendation is qualitative judgment, not a scored research result.

Results, Runs and Gallery are views over the same workstation,
not obligatory stages. Keep the model, queue and draft available across them.
Prototype their transitions before committing to their labels or placement.
The main risk is replacing today's many surfaces with one crowded surface.

Keep a credible alternative in the first prototype: a review-focused layout
with the composer immediately recoverable and its draft intact. Compare that
against the shared view on the same short tasks. Do not build three products.

## Established decisions constrain the comparison

- Preserve Generate, Batch, Repeat, explicit grid work and the existing seed
  semantics. Do not rename distinct operations to an ambiguous Continue.
- Keep **More seeds** separate from **Finish run**. More seeds joins the run;
  Finish addresses missing work rather than silently enlarging a completed run.
- Preserve accepted counts, previews, holds, undo and user-owned sampling
  values. A cleaner layout must not replace an owned value with a default.
- Keep image, batch, picked images, run and filter scopes distinguishable.
  A run action ignores Starred where already decided; picks retain their rules.
- Preserve the bin and permanent-delete distinction, source provenance,
  checkpoint restrictions and the server's authoritative eligibility answer.
- Keep stationary controls, focus return, draft persistence and meaningful
  refusal reasons. Background results must not move a destructive target.
- Do not introduce a project taxonomy, mandatory onboarding or collaboration.
  Matt's archive-and-clear workflow remains a first-class completion point.
- Leave external photo/sketch references deferred (`GEN#8`); they are not a
  prerequisite for testing the existing gallery-reference workflow.
- Leave Widen and Load into panel deferred under the accepted continuation
  plan. A comprehensive design exercise does not approve those features.

The [continuation plan](../plan-continue-run.md) remains the owner of
`GEN#11`. Its phase 1 exists; phase 2's retained cells and Finish run do not.
The design may depict that future state, visibly labelled as a prototype.
It must not promise restart recovery from current production behavior.

## One-user formative protocol

Use Matt's own task language and his actual desktop setup. The purpose is to
find and remove friction in this workstation, not estimate population rates.
Separate prior decisions, reported habits, observed behavior and hypotheses
in the record. An agent's walkthrough is an expert review, not a user trial.

Use a fresh fake-server gallery with synthetic prompts, images and runs.
Prepare complete and interrupted grids, two checkpoints, a refine source and
child, stars, binned images and selected subsets. Include a run larger than
the immediately visible view. No live gallery, encrypted bundle, private
prompt, weight download or real generation is needed for this comparison.

1. Record the current build, browser, CSS viewport, Windows scaling and input
   method. Note which relevant behaviors already exist in that build.
2. Run a short baseline on the current UI before teaching the proposed one.
   Give the desired outcome; avoid naming the button that would achieve it.
3. Ask Matt to predict scope and result before Queue on critical actions.
   Record his words, then let him proceed without correcting him early.
4. Compare low-fidelity alternatives with equivalent synthetic tasks. Change
   the prompts and seeds between versions; alternate order on repeated trials.
5. Ask what felt slower, uncertain or unexpectedly different. Keep quotes
   attributed and dated; do not convert preferences into observed task errors.
6. Revise the smallest part responsible for confusion and repeat that task.
   Keep unresolved tradeoffs visible rather than averaging them into a score.
7. After a viable prototype, assess it over several ordinary sessions with
   Matt. That later step needs a working implementation and normal-use evidence.

Repetition introduces learning. Record task order and prior exposure; avoid
claiming a time improvement caused by layout when familiarity could explain it.

## Core comparison tasks

| Task and synthetic setup | Evidence to collect |
|---|---|
| A Batch 4 finishes; ask for four more like it | Correct batch scope, default count, total and run identity; presses and detours |
| A keeper suggests a prompt edit while an unrelated draft exists | Clear Reuse effect, preservation or explicit replacement of draft, usable undo |
| Plan a multi-axis overnight run with a time budget | Ability to explain combinations, image count, estimate limits and stopping behavior |
| Reopen after a budget stop, with the composer changed | Recognition of missing work and original budget; no manual seed reconstruction |
| Finish after reload or restart in the future-state prototype | Correct original cells and run; existing work and queued work not duplicated |
| Starred is on; some images are picked; a whole run is open | Correct prediction of the scope of each action before Queue |
| Change checkpoint, then inspect an older image or run | Model state understood; held actions explain the mismatch; draft remains coherent |
| Compare two keepers and inspect a refine child against its source | Understand differences, source relationship and path back to the prior view |
| Open an action preview; change its underlying scope elsewhere | Stale preview recognized; recheck required; no silent change in queued work |
| Curate keepers, bundle, then perform normal cleanup | Clear retained scope, completed export verification and recoverable bin behavior |
| Repeat the short loop by keyboard in a short desktop viewport | Reachable actions, logical focus, intelligible state and no focus loss |

Recovery tasks need fault and legacy cases as well as the happy path: no
image made, source missing, older run without retained cells, another quantizer,
work still queued, all results binned and some permanently deleted results.
The expected answer may be a clear hold. Do not reward an action merely for
being available when the accepted rules require it to refuse.

## Baseline and provisional targets

**No baseline has been collected for this design.** Record task completion,
wrong-scope attempts, accidental submissions, lost context, surface changes,
navigation reversals, presses and assistance. Time supports these observations;
fake generation time is not part of interaction time. Keep raw observations.

These are provisional acceptance targets to review after the baseline:

- The accepted four-more path remains at most two presses from its finish
  action, with no picks, search or manual seed arithmetic required.
- Matt predicts scope and output count correctly in every critical scope
  trial before Queue. Any wrong-scope submission blocks that interaction.
- Future-state Finish after reopening requires no reconstruction of the
  original grid or seed, and does not read the unrelated current draft.
- Reviewing a run preserves the draft; deliberate replacement has useful undo.
- A changing model or preview never silently submits different work.
- No result arrival moves an actionable target under a stationary pointer.
- Critical tasks work by keyboard with visible focus and sensible return.
- At Matt's actual scaling and short viewport, progress and action controls
  are reachable without trapping the wheel or hiding refusal explanations.
- Archive completion is apparent, and the chosen image scope remains clear.

Do not publish percentages or broad usability claims from this one person.
Report which tasks passed, the context, assistance and unresolved problems.

## Accessibility and desktop checks

Use real keyboard events in a real browser, not only scripted focus calls.
Check Tab, Shift+Tab, Enter, Escape, image navigation and action shortcuts;
verify focus through open, cancel, submission, completion and deletion.
Exercise NVDA or Narrator with the actual UI: names, scope, held reasons,
progress announcements and focus return. Accessibility-tree inspection alone
does not complete the existing `UI#7` screen-reader work.

Check reduced motion and forced colours, actual Windows scaling, and short
desktop windows including 1536x864 where current progress limitations are
documented. Include 1400x900 and Matt's normal display configuration. A cropped
large screenshot does not demonstrate the behavior of a shorter viewport.

## Decision gates and implementation order

1. **Evidence gate:** Matt confirms the task priorities and unresolved
   assumptions; the current baseline and accepted behaviors are recorded.
2. **Concept gate:** compare the workbench and review-focused alternative;
   scope, draft continuity and make-judge adjacency must work before styling.
3. **Prototype gate:** critical tasks meet the provisional targets, or the
   targets and reasons for changing them are explicitly recorded with Matt.
4. **Implementation gate:** stage bounded slices using existing server
   capabilities first. Keep `GEN#11` as its own dependency and owner; connect
   its durable recovery only after phase 2 exists and is verified.
5. **Normal-use gate:** review several sessions before declaring the full
   redesign complete. Revisit unresolved observations in the design index.

Technical tests prove request semantics and failure handling. Browser smokes
prove specified UI states, layout, focus and interactions with fake data.
Matt's trials supply usability evidence. GPU timing, memory, pixels and model
qualification require separately authorized card work in `docs/verified.md`.
None of these forms of evidence substitutes for the others.

## Technical checks completed for this design, 2026-10-18

The isolated [concept](prototype/index.html) was checked in headless Edge
using the repository's `tools/browser.mjs`. The repeatable
[check script](prototype/check.json) passed all 34 expectations, covering:

- A draft retained through Results/Runs/Gallery and review focus.
- More seeds counts for a batch, whole run, picked batches and filter;
  hidden picks retained; explicit exclusion of a foreign checkpoint.
- Invalid totals held with a reason; Enter submits the default batch preview.
- Future Finish uses its run/budget; a queued run's Finish stays held.
- Reuse Undo restores the draft and retires after another edit; foreign
  Reuse carries a mismatch warning rather than switching the model.
- Image inspection stays open under More seeds; Escape and focus return
  operate on the top dialog; two picked images open a labelled comparison.
- No document-width overflow at 1400×900, 1536×864, 2048×1040 at DPR 1.25,
  and narrow CSS viewports of 390×844 and 320×720. Generate fits in the
  tested 1536×864 default concept state. These are CSS viewport checks,
  not actual phone/touch-device validation.
- Reduced-motion emulation and no captured runtime errors during the run.

Screenshots of the default, review-focused, action-preview and narrow
layouts were inspected. This found and corrected an aspect-ratio overflow
in the narrow Results view. A current-app screenshot was also inspected
using a scratch fake server; it is not a human-task baseline or a timing
comparison. No real gallery or weights were opened.

To repeat the checks from the repository root, run this in one terminal:

```powershell
uv run python -m http.server 7911 --bind 127.0.0.1 --directory docs/design
```

Then, in another terminal:

```powershell
node tools/browser.mjs docs/design/prototype/check.json
```

The concept has no persistence or application API. Its queue, eligibility
and counts are fixed synthetic fixtures, not implementations of production
logic. Native prompt Ctrl+Z, real model changes, stale server previews,
image painting, archive execution and restart durability are not implemented
in it. Those contracts remain specified in workbench.md and the existing
feature owners. No application Python or browser regression suite was run
because application code did not change.

No personal task trial, real screen reader, forced-color runtime check,
actual touch/phone test, Windows display-scaling session or GPU measurement
was completed. Screenshot/browser inspection is not evidence that the new
layout is faster for Matt. DEV#1 remains open for that evaluation.
