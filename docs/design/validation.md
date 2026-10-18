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

The fuller [feature-use response](preferences.md) now chooses the trial
priorities. It is evidence of reported use, not task performance or a blanket
rating for an area. All 129 listed features now have an explicit answer:
66 Often, 34 Sometimes and 29 Never. Grid as an area is Sometimes while
prompt variables specifically are Often; the other area ratings remain
unanswered. No further broad questionnaire is required to revise the next layout.

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
E11 rates Gallery beside Never: a simultaneous gallery/newest-result column
is not a required common-task layout. Keep session review and gallery review
direct, with full-size inspection and a contact sheet that can Fill width.

Keep a credible alternative in the first prototype: a review-focused layout
with the composer immediately recoverable and its draft intact. Compare that
against the shared view on the same short tasks. Do not build three products.

## Established decisions constrain the comparison

- Preserve Generate, Batch, Repeat, explicit grid work and the existing seed
  semantics. Do not rename distinct operations to an ambiguous Continue.
- Keep the frequent explicit Seed path and the confirmed Reset behavior:
  Reset clears Seed within its current scope and retains Undo; Matt enters
  another seed afterwards. Do not invent an empty-seed guard or auto-fill.
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
- Put progress, watched-job selection, queue/cancel, retry/remainder and
  reconnect ahead of unused Pause. Model guidance/favourites/preload take
  priority within Models; manual loader controls remain secondary.
- Test whole-gallery encrypted backup, archive-and-clear, download and
  import as common paths. Do not centre subset bundles, sharing or terminal
  workflows, all of which lack a supporting frequent-use answer here.
- Keep frequent Examples, guide/help, autocomplete, recommended Tags,
  chips and rules/order Check beside prompt editing. Occasional token
  review, corrections, critic/rewrite, shared-seed comparison, ERNIE Expand
  and Undo remain contextual; PNG settings is secondary. No helper applies
  advice automatically because it is frequent.
- Manual LoRAs/reference setup, time budgets and sheet export need no
  primary layout space. Recorded reference reuse (G3 Sometimes) remains
  a separate contextual action.

The [continuation plan](../plan-continue-run.md) remains the owner of
`GEN#11`. Check retained cells and Finish run against the exact current-app
build in the [trial kit](trials/README.md); phase 2 was built and verified
independently of this design. The concept's visibly labelled simulated Finish
does not establish current production restart recovery.

## Trial preparation

**Preparation built and verified, 2026-10-18 (`DEV#3`).** This can be done without Matt,
weights, a live gallery or a layout decision. `DEV#1` is the subsequent
hands-on trial, not a prerequisite for unrelated feature work.

Prepare a reproducible fixture and recording kit for the first short task
set: generation/Reset, busy queue/cancel/reconnect, inspection/sheet,
reuse/rerun and whole-gallery archive/import. Use disposable fake-server
data and the existing concept's synthetic data. Record their differences;
the concept's simulated Finish run is not production restart evidence.

Done means the kit includes:

- A repeatable seed/setup command with its own scratch `--data`, stable
  synthetic prompts/seeds, a complete and interrupted run, a source/child,
  stars, picks and bin contents. A second checkpoint can be planted without
  loading weights. State which cases the current app must refuse.
- Runnable current-app and concept launch/reset instructions, an equivalent
  task set with alternate values/order, and recorded build/browser/viewport
  and scaling. Do not simulate future APIs in the current-app baseline.
- An expert walkthrough of those tasks with real browser input, labelled
  as agent observations. Capture detours, scope and draft/focus behavior;
  leave Matt's timings, words and outcomes blank.
- A blank observation record for task order, prior exposure, predicted
  scope/count, presses, detours, assistance, outcome and feedback, followed
  by the unresolved layout choices. Do not select a redesign on his behalf.

The kit belongs beside this protocol; TODO remains the ordering/index.

The prepared [trial kit](trials/README.md) contains the disposable fixture
command, current-app/concept launch and reset instructions,
[alternate task cards](trials/task-cards.md), a
[blank human observation record](trials/observation-template.md) and a
separate [agent expert baseline](trials/expert-baseline-2026-10-18.md).
Preparation and agent checks do not complete `DEV#1` or supply human outcomes.

## One-user formative protocol

Use Matt's own task language and his actual desktop setup. The purpose is to
find and remove friction in this workstation, not estimate population rates.
Separate prior decisions, reported habits, observed behavior and hypotheses
in the record. An agent's walkthrough is an expert review, not a user trial.

Use a fresh fake-server gallery with synthetic prompts, images and runs.
Prepare complete and interrupted grids with prompt variables and Steps/CFG
sweeps, two checkpoints, a refine source and child, stars, binned images and
selected subsets. Include a run larger than the immediately visible view.
Archive screens use synthetic whole-gallery counts and a fake password,
progress and import result. No live gallery, real encrypted archive, private
prompt, weight download or real generation is needed for this comparison.
Fixed-fixture success checks the depicted interaction, not actual encryption,
server eligibility, archive verification or restart durability.

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
| Enter prompt and negative, select aspect/resolution, fixed Seed, Batch and Repeat; Reset and enter a new seed | Direct control reachability, distinct Batch/Repeat and total, values retained during review, visible cleared Seed and usable Reset Undo; no inferred preferred values |
| Insert an author example, consult guide/help, use tag autocomplete and recommended Tags, edit chips and Check rules/order | Helpers reachable beside the prompt; source/heuristic distinctions, undo, preserved newer text and meaningful applicability; no mandatory critic or Expand detour |
| Review a token/correction result or critic rewrite, optionally compare prompt versions on shared seeds or Expand with ERNIE | Context preserved, suggestion reviewed before replacement, usable Undo and controlled comparison labels; no assumption of automatic quality improvement |
| While a job runs, inspect earlier session results, queue another job and choose the job being watched | Progress/elapsed/estimate clarity, watched-job identity, draft retained and stable action positions |
| Cancel running and selected queued work, then retry skipped/remaining cells and reopen the tab | Correct target and outcome, reconnect to surviving work, saved results retained; no requirement to use Pause |
| A Batch 4 finishes; ask for four more like it | Correct batch scope, default count, total and run identity; presses and detours |
| A keeper suggests a prompt edit while an unrelated draft exists | Clear Reuse effect, preservation or explicit replacement of draft, usable undo |
| Plan prompt variables crossed with Steps/CFG and consecutive seeds | Access to variables, explanation of combinations/image count and optional estimates; no time-budget or explicit-seed-list detour |
| Reopen an interrupted run, with the composer changed | Recognition of missing work and the stored definition; no manual seed reconstruction |
| Finish after reload or restart in the future-state prototype | Correct original cells and run; existing work and queued work not duplicated |
| Starred is on; some images are picked; a whole run is open | Correct prediction of the scope of each action before Queue |
| Find older images by prompt, Stars, All models and Runs, then search from an image setting | Search scope and route back understood; picked set persists across filters; card-size/Fit/Crop controls reachable |
| Inspect full-size, zoom/pan, read prompt/settings, star and compare a refine child with its source | Direct inspection and source comparison; crop, inspected image and route back preserved |
| View the same run as a sheet and Fill width, then return to inspection | Useful image space and cell/settings orientation without exporting a sheet or using Gallery beside |
| Reuse settings or take/copy a seed, then use picked/filter More seeds, Re-render, Swap tags and Mass refine | Correct scope/counts and effective model eligibility; no lost picks/draft; frequent actions directly reachable |
| Refine/enlarge a keeper and try strengths/versions; open inpaint/outpaint when needed | Source, intended edit and output clear; common refine controls direct, occasional editors reachable without crowding review |
| Find model guidance/timing/examples, favourite a model, change startup preload and occasionally switch | Guidance discoverable, preload choice clear, model state understood; no manual loader-settings detour |
| Open an action preview; change its underlying scope elsewhere | Stale preview recognized; recheck required; no silent change in queued work |
| Back up the whole gallery with a password, archive-and-clear, download, then import a synthetic bundle | Whole-gallery scope apparent despite Stars/picks/model filter; clear follows verification, progress/result and import counts understood |
| Bin rejects individually or in bulk, clean unstarred in scope, restore a mistake and empty the bin | Fast ordinary binning, preserved picks, clear recoverability and confirmed permanent scope |
| Repeat the short loop using keyboard, right-click and Ctrl/Shift picks in a short desktop viewport | Reachable actions, logical focus, retained selection, intelligible state and no focus loss |

These are a prioritized task bank, not one long compulsory sitting. Start
with generation/reset, busy-queue control, inspection/sheet, reuse/rerun and
archive completion; split the rest into short sessions. Preserve occasional
inpaint/outpaint and grid paths without treating Never-rated features as
daily tasks. Arbitrary Compare picked remains an optional concept probe,
not evidence that F6's source comparison rated that future feature Often.

Recovery tasks need fault and legacy cases as well as the happy path: no
image made, source missing, older run without retained cells, another quantizer,
work still queued, all results binned and some permanently deleted results.
The expected answer may be a clear hold. Do not reward an action merely for
being available when the accepted rules require it to refuse.

## Baseline and provisional targets

**No human baseline has been collected for this design.** The separately
labelled [agent walkthrough](trials/expert-baseline-2026-10-18.md) is technical
preparation. In Matt's session, record task completion,
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
- Frequent generation controls, watched-job controls and inspection/sheet
  actions do not require visiting unused loader, LoRA, pause or export controls.
- Frequent prompt helpers are reachable beside prompt editing; occasional
  critic/rewrite/comparison/Expand tools do not displace the composer or
  automatically overwrite its newer text. Replacement Undo remains usable.
- Prompt variables remain directly reachable inside experiments, without
  requiring a time budget, explicit seed list or unused sweep axis.
- A whole-gallery archive remains the whole gallery when Stars, picks or a
  model filter is present; verification precedes clear and import reports its result.
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

**Separate completion boundaries, 2026-10-18.** `UI#9` is the ready browser
pass on today's app: use the synthetic fixtures, real key events and media
emulation for forced colours/reduced motion. Cover authoring and Check,
queue/cancellation, gallery picks, sheet, lightbox, one edit dialog and
archive/bin dialogs. Check visible focus, held reasons, focus return and
readable status without relying on colour or animation alone. Reuse the
existing focused smokes; add only missing cases. Record the build, surfaces,
viewports, media settings and failures here or in a linked review, and index
any confirmed defects separately. A successful pass records its bounded
coverage; it does not close the manual work.

`UI#7` needs a hands-on NVDA/Narrator and touch session with the actual
interface. Record the assistive technology/device and observed names,
announcements, operation and focus. Browser attributes, emulated media or
an accessibility-tree dump cannot supply that evidence. Neither row
authorizes broad layout changes; `UI#8` keeps its deferred scope.

**Baseline browser gaps observed, 2026-10-18.** At 600×900, the stacked
7:4 lightbox comparison gives its facts 196 px for 202 px of content.
`smoke-lightbox` retained that failing assertion; baseline `27b3625` web
assets served against the same disposable fake gallery reproduce identical
geometry. The new review actions passed their scoped tests. A bin workflow
also exposed an existing viewer tombstone extending to x=613.5 at 600 px;
the bin's picked-delete controls themselves fit. These narrow-view gaps
remain separate evidence for `UI#9`, not completed screen-reader/touch work.

**Browser pass completed, 2026-10-18 UTC (`UI#9`).** The immutable
`0604f5a` application with the final accessibility CSS and smoke changes
overlaid passed seven full smokes: Lightbox 229, Sheet 205, Select 156,
Bin 112, Bundles 117, Analysis 231 and Preview 20: **1,070/1,070**
expectations, including 78 new checks. Each used a fresh scratch gallery
and an owned fake server, sequentially. Coverage includes normal and
forced-colour/reduced-motion modes, 1400x900, 600x900, 1536x864, QHD and
emulated DPR 1.25. Real Tab/Shift+Tab, Enter, Space, Escape and arrow
events exercise authoring/Check, queue/cancel, picks, sheet, lightbox,
Refine and archive/bin dialogs. Focus checks require the element to be
focused, focus-visible, on screen and unobscured; status and held reasons
remain readable without animation or colour alone.

Scoped fixes add forced-colour rings to fields, Check links, menus,
Refine thumbnails and the comparison divider, plus underlines for pressed
comparison modes. The viewer tombstone now stays within the narrowed
viewport while retaining its measured height. The 196px/202px lightbox
facts measurement above reflects the intentionally capped scroller, not
a requirement that metadata never scroll: 15 trusted Tab presses reach
the final Run fact while image geometry stays unchanged. Visual inspection
confirmed readable forced-colour focus and selected-mode feedback.

Evidence: temporary `ld-ui9-final-e_e619fg/` contains the five other final
full smokes, screenshots and tested-file SHA256 manifests;
`ld-ui9-preview-final-prgaivbt/preview/` has corrected Preview 20/20;
`ld-ui9-bundles-final-d5pw5yv0/` has final Bundles 117/117. Earlier logs
retain the invalid Sheet setup fetch, transient Bundles fixture failures,
and corrected test navigation/literals (Shift+Arrow for the divider and
U+2026 in the spoken denoising status). No Bundles product defect or 409
cause was proved; fresh diagnostic and full reruns passed the unchanged
archive assertions. JSON/embedded-JavaScript parsing and scoped diff
checks passed. All owned fakes stopped; only verified owned leftover
headless browser trees were closed, with profile/time/target manifests.

This establishes the stated browser behaviour. NVDA/Narrator, touch, native
notification delivery and actual Windows display scaling remain manual
checks; `UI#7` is still open. It does not approve the candidate redesign.

## Decision gates and implementation order

1. **Evidence gate:** use Matt's recorded feature priorities and Reset
   decision; record the current baseline and accepted behaviors. Ask only
   about an unresolved detail that changes the next layout decision, not
   for another exhaustive feature rating.
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

### Reported control use follow-up

The fuller [self-report](preferences.md) supplies placement priorities,
not a completed task trial. The next revision should check:

1. Use separate Aspect ratio and Resolution presets and a clearly fixed
   sample Seed. Keep Batch and Repeat direct and distinct; show the total.
2. Place Steps/CFG/Mu in one secondary disclosure with their effective
   values readable. Keep never-used settings available secondarily.
3. Preserve the confirmed Reset behavior: clear Seed with the existing
   reset scope and retain Undo. Check that the empty seed is visible and
   another seed can be entered afterwards; do not auto-fill it or add a
   generation guard. Preserve direct access and draft-value ownership.
4. Keep job progress, watched-job selection, running/queued cancel and
   retry/remainder prominent; leave Pause secondary. Distinguish reconnect
   to a live server from future restart recovery.
5. Make full-size inspection, sheet/Fill width, search/Stars/All models,
   retained picks, Reuse/More seeds and selected-scope rerun/refine paths
   reachable without Gallery beside or unused sheet-export controls.
6. Put whole-gallery backup/password/archive-and-clear/download/import
   together, with explicit scope and truthful simulated states. Keep model
   guidance/favourites/startup loading ahead of advanced loader options.
7. Keep prompt variables prominent in experiments; test occasional
   Steps/CFG and consecutive-seed work without promoting unused budget,
   list, LoRA or Mu sweep controls. Keep inpaint/outpaint contextual.
8. Keep Examples, guide/help, autocomplete, Tags, chips and Check beside
   prompt editing. Test contextual token details, corrections, critic/rewrite,
   shared-seed comparison, Expand and Undo without promoting unused PNG
   settings or applying advice automatically.
9. Compare adjustment paths against today's app and the initial concept;
   record presses, wrong targets, unnecessary scrolling and whether the
   selected seed is apparent before Generate.
10. Retain unanswered area ratings as unanswered; all listed individual
    features are now rated. Frequency answers alone do not validate the
    whole-app navigation, image-space trade-offs or speed.

The initial concept and its 34 checks predate these preference revisions.
They do not qualify Repeat/Reset, the selector/sampling layout or the new
prompt/job/review/archive/model emphasis. Any revised prototype checks must be
reported separately with their actual result and fixture limitations;
passing them will still not establish personal speed or real application behavior.

### Preference-led prototype verification

**2026-10-18:** the revised [concept](prototype/index.html), including its
`composer.js` and `workflow.js` declarations, passed all **54 expectations**
in the current [check script](prototype/check.json) using headless Edge.
The first baseline checks are retained and extended, not presented as a
human usability trial. The checks cover:

- Independent aspect/resolution, effective sampling disclosure, readable
  sample Seed and Batch × Repeat totals.
- Reset clearing prompt/negative/Seed and restoring the represented WAI
  defaults; whole-draft Undo survives the programmatic focus/change events
  and later edits retire the earlier snapshot.
- The direct prompt-help routes without fabricated tokenizer findings or
  author examples; no helper runs or modifies text automatically.
- Watched-job switching and cancellation of a queued fixture, retaining
  the unrelated draft and saved sample images.
- Picked/filter action scope, cross-filter picks and More seeds preview
  counts; the existing future-Finish holds and dialog/focus checks remain.
- Query-based contact-sheet scope independent of picks, source comparison,
  and **actual cell enlargement** from Fill width at 1400 and 2048 CSS px.
- Whole-gallery archive scope despite picks; visible password/backup/clear/
  download/import routes, with explicit simulation feedback and no file
  mutation; model favourites/startup controls ahead of collapsed loader details.
- No document-width overflow for the default Results views at the earlier
  desktop/narrow sizes and for an overflowing sheet at 320 and 390 CSS px:
  its own table scroll contains the wide matrix. This is not phone validation.

The default composer, narrow layout, sheet and archive dialog were visually
inspected. Two important limitations remain: several direct routes demonstrate
placement/scope rather than execute production helpers or planners, and the
two job fixtures are independent of the illustrative run-outcome fixtures.
There is no real reconnect/restart engine in this concept.

The concept models only the represented fields of Reset: it does not clear
the application's conditioning/grid/editing persistence or implement its
Undo expiry. Native text writes attempt to retain browser Undo, but full
Ctrl+Z history, keyboard/right-click task parity, real server previews,
tokenizers, chips/autocomplete, model loading, painting and archive execution
have not been qualified here. Those remain application/feature-owner work.

All 129 captured labels and ratings were also compared with the latest
attachment and matched exactly; local Markdown links/anchors were checked.
Application code did not change. No application regression suite, live
gallery, model download, GPU run or actual personal task trial was required
or performed. Task times and speed improvements remain unmeasured.

### Initial concept verification

At the initial baseline (commit `d6957dc`), the isolated concept was checked
in headless Edge using the repository's `tools/browser.mjs`. The original
version of the [check script](prototype/check.json) passed 34 expectations;
the current script extends it as recorded above. That baseline covered:

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
in that initial baseline. Those contracts remain specified in workbench.md and the existing
feature owners. No application Python or browser regression suite was run
because application code did not change.

No personal task trial, real screen reader, forced-color runtime check,
actual touch/phone test, Windows display-scaling session or GPU measurement
was completed. Screenshot/browser inspection is not evidence that the new
layout is faster for Matt. DEV#1 remains open for that evaluation.
