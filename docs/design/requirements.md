# Candidate design requirements

2026-10-18. Scope: Matt's personal AuraDream workstation on the RTX 5080,
including its Windows and WSL environments. These requirements describe his
whole working cycle; they do not propose a public service, accounts or a new audience.

Read [research](./research.md), [workbench](./workbench.md) and
[validation](./validation.md) alongside this catalogue. The
[usage guide](../usage.md) describes today's app; existing plans and
[TODO](../../TODO.md) retain ownership of implementation and its ordering.

**Status reconciliation, 2026-10-28:** the current-capability references below
include the completed model-cache facts, Expand cancellation, restart recovery,
picked comparison/navigation and import work. The workbench layout and its
human task trials remain candidates; these source updates do not validate it.

`Current` means the behavior exists, not that usability is established.
`Accepted-unbuilt` identifies an existing decision awaiting implementation.
`Proposed` is a candidate for design and evaluation, not approval to build.
Evidence is repository behavior or Matt's recorded decisions; a hypothesis
requires checking with Matt. P0 protects task completion and work; P1 improves
judgment and continuity; P2 is conditional convenience. Priorities here do not
change TODO's NOW, authorize GPU work or activate deferred features.

**Leading outcome:** Matt explicitly asked to do common tasks quickly.
Preserve direct paths, count detours and compare task time against the current
app. The [fast-path table](workbench.md#fast-paths) defines provisional action
budgets. The fuller [feature-use response](preferences.md) now supplies personal
priorities across generation, experiments, jobs, review, editing, archives and
models. The complete response rates all 129 listed features: 66 Often,
34 Sometimes and 29 Never. These are self-reports, not measured time savings.
The broad Grid rating of Sometimes does not override its individual feature
answers; the other area ratings remain unanswered.

## Reported-use placement priorities

These placement decisions adapt the existing requirements; they do not make
every current capability a prominent control or remove unused capabilities.
An Often rating does not prescribe a default value or authorize automatic work.

| Placement to evaluate | Reported evidence | Requirements |
|---|---|---|
| Direct composer with explicit Seed, Batch, Repeat, aspect/resolution and Reset | A1–A4, A11–A14 Often; Steps/CFG/Mu Sometimes | UX#21 |
| Examples, prompting guide/help, autocomplete, recommended Tags, chips and Check beside prompt editing | B1–B5 and B7 Often; token counts, corrections, critic/rewrite, shared-seed comparison, Expand and Undo Sometimes; PNG settings Never | UX#4, UX#5, UX#9 |
| Persistent job status and direct job selection, cancel and retry/remainder actions; reconnect clearly | D1, D3–D5, D7–D10 Often; preview Sometimes, pause Never | UX#1, UX#7, UX#10 |
| Direct gallery search, Stars, All models, Runs, picks and density controls; full-size inspection and a contact sheet with Fill width | E1–E5, E9–E10, E12 and F1–F3, F5–F6, F9, F11 Often; Gallery beside Never | UX#9, UX#11, UX#12, UX#19 |
| Reuse settings, take/copy seed or prompt, More seeds, Re-render, Swap tags and Mass refine at the image or selected-scope action point | G1–G2, G4–G9, H1–H2, H4–H5, J1 Often | UX#5, UX#6, UX#11, UX#14 |
| Prompt variables prominent within experiments; Steps/CFG and consecutive-seed sweeps available there | C7 Often; C1–C2, C5, C8–C9, C11 Sometimes; other C controls Never | UX#8 |
| Whole-gallery backup, password protection, archive-and-clear, saved downloads and import directly reachable from gallery completion | J9, J11–J14 Often; subset bundles Never | UX#16, UX#17, UX#18 |
| Model guidance, timing/example references, favourites and startup preload ahead of loader adjustments | K2–K3, K8 Often; switching and WSL Sometimes; K4–K7 Never | UX#2, UX#3 |
| Keyboard, image right-click actions and Ctrl/Shift selection remain first-class paths | L6–L8 Often; terminal tools L1–L5 Never | UX#12, UX#20 |

Inpaint/outpaint and their controls remain contextual Sometimes tools. Manual
LoRAs and reference management, time budgets, sheet export, clipboard image
copy, original-PNG opening and advanced loader controls stay accessible
secondarily. G3's Sometimes reuse of recorded references remains reachable
without promoting manual reference setup, rated Never. PNG settings remains
secondary; occasional critic, rewrite, comparison and Expand stay reachable
in their prompt context. See [preferences](preferences.md) for the complete
answers and exceptions.

## Workstation preparation

### UX#21 — Reach my frequent generation controls directly · P1 · Proposed

- **Basis:** direct self-report: A1–A14 in [preferences.md](preferences.md), 2026-10-18. Frequently used controls differ from occasionally adjusted sampling and never-used options.
- **Story:** when I am generating, I can edit my prompt and negative, choose aspect/resolution, set a fixed seed and Batch/Repeat, or Reset without opening secondary controls.
- **Acceptance:** the next composer study shows separate aspect/resolution selectors, a readable fixed Seed, distinct Batch and Repeat with their total, and a direct undoable Reset. Specific values remain owned or illustrative, never inferred from frequency.
- **Acceptance:** Steps/CFG/Mu are one disclosure away with effective values visible; unused custom dimensions/orientation, True CFG and random-seed actions remain accessible secondarily. No unreported feature is hidden on the basis of these answers.
- **Trace:** [placement](workbench.md#reported-generation-control-priorities), [validation](validation.md#reported-control-use-follow-up), `DEV#1`. Matt confirmed today's Reset behavior: clear Seed, retain existing reset scope and Undo, then let him enter another seed. No automatic replacement or new empty-seed submission guard.
- **Dependency:** model applicability and existing persistence/undo; no application implementation is implied by this design requirement.

### UX#1 — Find the next useful action · P1 · Proposed

- **Basis:** D1, D3–D5 and D7–D10 Often supports direct orientation to progress, session results and queued work. The particular layout remains a hypothesis; current generation, results and gallery already share one page.
- **Story:** when I return to my workstation, I can tell what is ready, what is running and where to resume judging or creating.
- **Acceptance:** the workbench concept identifies the running checkpoint, active/queued work, watched job, earlier session results and selected-image scope without opening unrelated dialogs. A paused state remains legible when present without giving Pause primary space.
- **Acceptance:** from cold, busy and returning-session fixtures, Matt can name the next action for his task before pressing it; record confusion rather than assuming a new navigation scheme is needed.
- **Trace:** [usage: jobs](../usage.md#generate-and-manage-jobs), [existing modes](../plan-continue-run.md#personas); no existing TODO item claims this full-app orientation outcome.
- **Dependency:** research's task priorities; preserve the existing no-build-step workstation architecture.

### UX#2 — Know which model is selected and usable · P0 · Current

- **Basis:** evidence: Models distinguishes selected, loaded, loading and failed states, saved choices and unsaved changes. K2–K3 and K8 are Often; switching is Sometimes and manual loader adjustment/actions are Never.
- **Story:** I can find model guidance and favourites, choose startup loading, and occasionally switch checkpoints while understanding the effective model state.
- **Acceptance:** cold start, save without load, successful load, failed load and another-tab switch each show the effective checkpoint/options and an available next action.
- **Acceptance:** checkpoint changes update supported controls and defaults; a stale dialog or submission cannot silently use the previous model's assumptions.
- **Acceptance:** model guidance, favourite status and Load automatically on startup are direct within Models; quantization, offload, save-without-loading and manual load/unload remain accessible secondarily. Failures still expose an intelligible recovery action even though manual loading is not a routine task.
- **Trace:** [usage: models](../usage.md#load-and-switch-models), [switching owner](../design-notes.md#switching-checkpoints-while-the-server-stays-open); `GEN#1`'s offline cache/download facts are built, recorded in [F7](../review-2026-10-12.md#f7-show-each-checkpoints-download-size-and-whether-it-is-already-in-the-hf-cache).
- **Dependency:** current model revision and load lifecycle rules remain authoritative.

### UX#3 — Judge preparation cost before committing time · P1 · Proposed

- **Basis:** K2 Often supports prominent model guidance, recorded timings and examples. First-use downloads and reference timings are documented; Models already shows offline cache/download facts (`GEN#1`, built). Their placement in a redesigned workbench remains a hypothesis.
- **Story:** I can distinguish a ready local choice from one that may need downloads, approval or unmeasured resource use.
- **Acceptance:** a candidate model row separates cached/download information where known, access requirements, recorded speed references and unmeasured combinations; unknown values remain explicit.
- **Acceptance:** a reference timing is never presented as an ETA for my prompt or as proof that a size fits the card.
- **Trace:** [usage: models](../usage.md#load-and-switch-models), [verified measurements](../verified.md); `GEN#1` is built, while remaining card measurements are indexed in TODO. A switch's live access preflight is built ([Switching](../design-notes.md#switching-checkpoints-while-the-server-stays-open), 2026-10-19), while the rows' access labels stay the static catalogue word.
- **Dependency:** use existing sourced facts and local cache inspection; no model download or benchmark is required to explore the design.

## Prompting and immediate iteration

### UX#4 — Understand prompt advice and its limits · P0 · Current

- **Basis:** B1–B5 and B7 Often supports author examples, guide/help, autocomplete, Tags, chip editing and rules/order Check beside prompt editing. B6, B8–B12 and B14 Sometimes supports contextual token review, corrections, critic/rewrite, comparison, Expand and Undo; PNG settings (B13) is Never. Advice retains distinct provenance and confidence.
- **Story:** I can decide which advice to use without mistaking a suggestion for a rule or an image-quality guarantee.
- **Acceptance:** documented guidance retains its source; heuristics and unchecked/unavailable results are identified; the critic's rewrite remains reviewable before adoption.
- **Acceptance:** inactive negative guidance, literal weighting syntax, network tag lookup and cold Check behavior are understandable in their relevant controls.
- **Acceptance:** Examples, guide/help, autocomplete, recommended Tags, chips and Check are directly reachable beside the prompt; autocomplete and chip edits preserve keyboard access and text undo. Token/truncation details, corrections, critic/rewrite, shared-seed comparison and ERNIE Expand remain contextual secondary actions; PNG settings is secondary. Prominence never implies that advice has been applied automatically.
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [analyser owner](../design-notes.md#the-prompt-analyser); no new TODO row; `PA#9` automatic image scoring stays deferred.
- **Dependency:** checkpoint-specific guidance and current validation boundaries.

### UX#5 — Recover my text and settings after replacements · P0 · Current

- **Basis:** G1 Often and B1 Often confirms frequent Reuse and example insertion; corrections, critic/rewrite, Expand and Undo (B8–B10, B12, B14) are Sometimes. PNG settings (B13) is Never. All existing replacement paths still need their current undo protection.
- **Story:** I can try advice or reuse an image without losing the prompt and controls I was editing.
- **Acceptance:** each replacement uses the established text undo path; Reuse/PNG restoration offers the documented settings Undo and names replaced prompt text and omitted inputs.
- **Acceptance:** edits made while Expand is waiting survive its answer; stale advice cannot silently overwrite newer text.
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [PNG settings](../usage.md#load-settings-from-a-png), [Reuse owner](../design-notes.md#per-control-defaults-and-reuse); Expand cancellation (`GEN#2`) is built, recorded in [Stop Expand](../design-notes.md#stop-expand-2026-10-18).
- **Dependency:** preserve separate prompt-text undo and whole-settings restoration semantics.

### UX#6 — Iterate from the result I am judging · P0 · Current

- **Basis:** G1–G2, G4–G6 and J1 Often confirm frequent reuse, seed retrieval, batch/run More seeds and prompt/seed copying. Continuation phase 1 is built on the finish line, image menu and lightbox.
- **Story:** when a result is promising, I can make more of its batch or reuse its settings while still judging it.
- **Acceptance:** batch and run entry points identify their target, images per cell, total work and seed policy before Queue; repeated submissions skip already planned seeds.
- **Acceptance:** lightbox/dialog Escape and return focus preserve the inspected image; narrow-width access remains available through the existing image menu.
- **Acceptance:** Reuse settings, take Seed, copy prompt/seed and More seeds are directly reachable from inspection or the image menu. Recorded-reference reuse (G3 Sometimes) remains available contextually; its use does not imply manual reference management (I3–I4 Never).
- **Trace:** [usage: continue](../usage.md#continue-a-run), [accepted continuation design](../plan-continue-run.md#the-recommended-design); phases 1 and 2, including `GEN#11`'s persisted Finish run, are built.
- **Dependency:** preserve batch settings and joining the original run; a new continuation shortcut is not required.

### UX#7 — Understand work while it runs or waits · P0 · Current

- **Basis:** D1, D3–D5 and D7–D10 Often prioritizes progress, session results, queueing, job selection, cancel, retry/remainder and reconnect. Live preview is Sometimes; pause/resume is Never. Existing states still have distinct behavior.
- **Story:** I can tell what the workstation is doing, choose the job I am watching, queue or cancel work, retry its remainder and return later without discarding saved results.
- **Acceptance:** representative loading, queued, paused, canceling, failed and completed states identify the work and relevant available action; a held action explains its reason when pressed.
- **Acceptance:** closing/reopening a tab reconnects to surviving server work; a server restart does not imply that queued jobs survived.
- **Acceptance:** progress/elapsed/estimates, watched-job identity, running and queued cancel actions, and retry/remainder remain direct without opening advanced job controls. Live preview can be shown contextually; Pause is secondary and retains its existing semantics. New arrivals do not displace the watched job or move its cancel target.
- **Trace:** [usage: jobs](../usage.md#generate-and-manage-jobs), [queue owner](../design-notes.md#the-browsers-queue); notifications (`UI#2`) and Expand cancellation (`GEN#2`) are built. Actual native notification delivery remains a hands-on check under `UI#7`.
- **Dependency:** single-worker lifecycle and cooperative cancel/pause contracts; no background auto-queueing.

## Experiments, comparison and continuation

### UX#8 — Predict a grid before running it · P0 · Current

- **Basis:** Grid as an area is Sometimes, while prompt variables (C7) are Often. Steps/CFG sweeps, consecutive seeds, crossed axes, preview and resume are Sometimes; Mu/LoRA sweeps, explicit seed lists and time budgets are Never. The preview already resolves cells and estimates before submission.
- **Story:** I can plan a bounded experiment and understand how many images it will create before using the card.
- **Acceptance:** prompt variables are immediately available inside explicit experiment planning. A fixture crossing Steps/CFG, consecutive seeds and prompt variables shows resolved prompts, total images, existing/queued cells and invalid or duplicate effective settings. Mu/LoRA sweeps, explicit lists and time budgets remain secondary rather than prerequisites to ordinary experiments.
- **Acceptance:** budget wording describes starting no further cell, and unavailable estimates remain unavailable rather than inferred from unrelated measurements.
- **Trace:** [usage: grid](../usage.md#sweep-settings-with-grid-search), [preview owner](../design-notes.md#grid-preview-and-the-per-step-estimate); no new TODO row.
- **Dependency:** server expansion and normalization stay authoritative; existing image and batch guards remain intact.

### UX#9 — Compare a controlled experiment intelligibly · P0 · Current

- **Basis:** F1–F3, F5–F6, F9 and F11 Often prioritizes full-size inspection, settings, stars, source comparison, contact sheets and Fill width. CFG comparison, comparison modes, sheet axes and shared-seed prompt comparison (B11) are Sometimes; sheet export is Never. Same-seed comparisons expose controlled differences without proving image quality from a small sample.
- **Story:** I can judge the effect of a change and understand what differed between the images.
- **Acceptance:** prompt versions remain identified within their submission, shared settings and changing axes are visible, and source/refine comparisons retain a meaningful crop.
- **Acceptance:** sheets and exported labels describe their scope; a small seed sample or a rewrite changing multiple things is not presented as conclusive evidence.
- **Acceptance:** full-size navigation/zoom, full prompt/settings, star, source comparison, contact-sheet entry and Fill width are direct in their review contexts. Sheet axes/slices and CFG/mode controls remain contextual; export is secondary. Gallery beside is not required to reach them (E11 Never).
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [usage: gallery](../usage.md#browse-compare-and-keep-results), [sheet owner](../design-notes.md#the-contact-sheet); Seeds to grid (`UI#3`) and sheet keyboard navigation (`UI#4`) are built.
- **Dependency:** distinguish same-seed experimental comparison from arbitrary aesthetic preference.

### UX#10 — Finish interrupted work after a restart · P0 · Current

- **Basis:** continuation phase 2 is built: admitted jobs retain resolved cells and terminal outcomes, and Finish run recovers missing work after a restart without the panel's grid.
- **Story:** the morning after a restart, I can finish a stored run without reconstructing its axes or base seed.
- **Acceptance:** Finish is held while that run has queued work, naming the job that holds the rest. Otherwise it previews and queues only missing, unplanned cells from stored jobs under the original run identity; binned cells are not redrawn.
- **Acceptance:** legacy/imported runs, changed model/quantizer, missing sources/adapters/masks, deleted results and storage failures follow the accepted phase-2 refusals and disclosures.
- **Trace:** [phase-2 owner](../plan-continue-run.md#phase-2-runs-keep-their-cells), [accepted answers](../plan-continue-run.md#open-questions-for-the-user); `GEN#11` is complete, independently of whole-app layout validation.
- **Dependency:** preserve persisted job definitions/outcomes and ordinary submission checks; legacy/imported runs without retained definitions remain explicitly unavailable for Finish.

### UX#11 — Predict the scope of every image action · P0 · Current

- **Basis:** evidence: picks, filter, whole run, all-checkpoint Starred and whole gallery are deliberately different scopes.
- **Story:** before I act, I know which images are included, including those hidden by the current view.
- **Acceptance:** Re-render, Swap tags, Mass refine and footer More seeds identify picks versus filter; run continuation identifies the whole run and ignores Starred as documented.
- **Acceptance:** bundle scope explains its all-checkpoint behavior; previews report counts, exclusions and resulting images without silently changing the target.
- **Trace:** [usage: selection/actions](../usage.md#browse-compare-and-keep-results), [bundles](../usage.md#gallery-bundles), [selection owner](../design-notes.md#the-gallery-selection); no new TODO row.
- **Dependency:** retain deliberate scope semantics; a consistent presentation pattern is evaluated in the workbench, not an assumed backend unification.

### UX#12 — Curate a set across searches without losing it · P1 · Proposed

- **Basis:** E1–E5, E9–E10, E12 and G9 Often supports browsing, Stars, All models, Runs, prompt/from-setting search, card sizing/Fit/Crop, retained picks and bulk starring. Persistent picks/hidden counts, Ctrl+A and Pick this run are built (`UI#6`); their placement and clarity remain part of task trials.
- **Story:** I can gather keepers across my runs, inspect them and apply an action to exactly that set.
- **Acceptance:** design trials retain picks and anchor across search/paging and a bin visit; hidden-pick counts remain understandable and Clear is deliberate.
- **Acceptance:** existing pick-all/run controls state their loaded/run scope and cap before action; they do not quietly select an unlimited gallery.
- **Acceptance:** browse/older images, prompt search, Stars, All models, Runs, density/Fit/Crop and picked-star actions are directly available in gallery review; searches from an image's prompt/settings preserve the relevant route back. Technical search syntax remains available, while LoRA/source/mask/edit-type search (E8 Never) is secondary.
- **Trace:** [usage: gallery](../usage.md#browse-compare-and-keep-results), [selection owner](../design-notes.md#the-gallery-selection); `UI#6` and a sheet of picks are built. This requirement evaluates the curation journey rather than creating another implementation row.
- **Dependency:** UX#11; keep current per-card star behavior and the separate temporary bin selection.

### UX#13 — Compare two chosen candidates · P1 · Current

- **Basis:** Compare picked (`UI#1`) is built alongside source/pass and CFG-neighbor comparison. F6 Often and F7–F8 Sometimes rates the original comparison features; those answers do not measure the usefulness of arbitrary pairs.
- **Story:** when I have two candidates, I can inspect their visual and recorded-setting differences directly.
- **Acceptance:** Compare picked opens exactly two picked images with matching aspect ratios, labels the pair and differing settings, and supports the existing fit/zoom and comparison modes.
- **Acceptance:** missing/binned inputs and an unsupported picked count have explicit recoverable outcomes; closing comparison returns to the retained selection and focus.
- **Trace:** [usage: lightbox](../usage.md#keyboard-and-lightbox), [comparison owner](../design-notes.md#the-lightbox-zooms), `UI#1`.
- **Dependency:** UX#9, UX#11 and UX#12; preserve the implemented interaction and evaluate its placement in any candidate layout.

## Editing, retention and recovery

### UX#14 — Choose the edit that preserves what I intend · P0 · Current

- **Basis:** H1–H2, H4–H5 Often and G7–G8 Often supports direct Refine/enlargement, versions/strengths, Mass refine, Re-render and Swap tags. Changed-prompt refine and inpaint/outpaint are Sometimes; cross-model/existing-pass mass refine is Never. Existing edits retain distinct meanings.
- **Story:** I can improve a keeper while understanding which parts or composition may change.
- **Acceptance:** representative edit dialogs identify their source, output size, strength, versions, supported checkpoint and relevant measured/unmeasured guidance before running.
- **Acceptance:** Inpaint previews its painted mask and Outpaint its padding/band; held capabilities explain why and preserve keyboard access to that explanation.
- **Acceptance:** frequent Refine/enlarge and strength/version controls are direct in the image action flow; Mass refine, Re-render and Swap tags are directly available at the picked/filter scope point. Inpaint/outpaint remain easy to open as contextual tools; cross-model/existing-pass inclusion remains secondary without changing defaults or eligibility.
- **Trace:** [Refine](../usage.md#refine-an-existing-image), [Inpaint](../usage.md#inpaint-part-of-an-image), [Outpaint](../usage.md#outpaint-an-image), [edit plans](../plan-inpainting.md), [outpaint plan](../plan-outpainting.md); `INP#5`–`INP#10` retain their own status (FLUX.1 inpaint, `INP#3`, was enabled 2026-10-19).
- **Dependency:** current size guards and measured families; personal-file references `GEN#8` remain deferred until Matt raises them.

### UX#15 — Retain an edit's lineage and reproduce its intent · P0 · Current

- **Basis:** evidence: saved images retain sources, masks and outpaint specs; source removal affects reproducibility; Mass refine intentionally drops an inpaint mask.
- **Story:** I can trace a result back to its inputs and recognize when a rerun can no longer reproduce the original operation.
- **Acceptance:** facts, Reuse and rerun previews identify source and mask/spec; a binned source offers the documented recovery and a permanently missing one is not silently replaced.
- **Acceptance:** inpaint reruns preserve source size/mask, outpaint reruns preserve padding/fill/band, and an action that drops a mask states that resulting operation.
- **Trace:** [usage: edits](../usage.md#inpaint-part-of-an-image), [inpaint reproduction](../plan-inpainting.md#the-reproduction-set), [outpaint lineage](../plan-outpainting.md#lineage-and-gallery-actions); related measurement rows `CARD#1`–`CARD#3` remain open.
- **Dependency:** UX#14 and stored reproduction invariants; browser checks cannot establish pixel quality or GPU fit.

### UX#16 — Back up, archive and clear the intended gallery · P0 · Current

- **Basis:** J9 and J11–J14 Often prioritizes whole-gallery backup, password protection, archive-and-clear, saved downloads and import. Subset bundles (J10) are Never; existing subset semantics remain intact without implying a sharing need.
- **Story:** I can back up the whole gallery, protect the bundle with a password, download it or clear the working gallery after verified export, then import it later.
- **Acceptance:** whole-gallery backup, subset backup and bundle-and-clear distinguish retained/deleted contents, scope, omitted links and encryption behavior before action.
- **Acceptance:** only verified export permits clear; failures retain a reviewable result, and archive progress remains recoverable after closing the browser.
- **Acceptance:** whole-gallery backup, password entry, bundle-and-clear, saved download and import are direct within the archive flow. Its whole-gallery scope remains apparent even with picks, Starred or a model filter active; unused subset export remains secondary and is never silently selected.
- **Trace:** [usage: bundles](../usage.md#gallery-bundles), [bundle owner](../design-notes.md#gallery-bundles-verify-before-clearing-stage-before-merging); `GAL#3`'s bounded CPU scale/interruption checks are complete in the [storage review](../review-2026-10-12.md#storage-scale-and-interruption-checks).
- **Dependency:** gallery maintenance admission and existing verification/staging; no automatic inclusion of unchosen images.

### UX#17 — Find and understand restored work · P1 · Current

- **Basis:** J14 Often confirms import/merge as part of the working cycle. Import preserves/remaps records and reports outcomes; Show imported (`GAL#2`) is built. A broader archive organizer remains a hypothesis.
- **Story:** after importing an archive, I can inspect the images just added and tell which dependencies were recovered or omitted.
- **Acceptance:** import results distinguish added, skipped and restored images, omitted links and failures; a Show imported route reaches the returned imported-image scope without manual id reconstruction.
- **Acceptance:** importing duplicate content preserves the documented current-gallery star/bin state; an empty import provides an intelligible result.
- **Trace:** [usage: bundles](../usage.md#gallery-bundles), [bundle owner](../design-notes.md#gallery-bundles-verify-before-clearing-stage-before-merging); `GAL#2` and the bounded CPU checks for `GAL#3` are complete.
- **Dependency:** server import identity/range answer and UX#11; permanent archive browsing or project management needs separate evidence.

### UX#18 — Recover deletion and understand permanent loss · P0 · Current

- **Basis:** J4–J6 and J8 Often prioritizes individual/bulk binning, scoped unstarred cleanup and permanent cleanup; restore (J7) is Sometimes. Existing immediate binning with Restore and permanent confirmation remain authoritative.
- **Story:** I can discard rejects quickly, recover mistakes and recognize which permanent deletion would prevent later reproduction.
- **Acceptance:** single/bulk binning offers Restore, bin search stays within the bin, and visiting it preserves gallery picks; source dependencies remain visible.
- **Acceptance:** permanent actions confirm their actual scope and lineage consequence; changed contents cannot be removed under a stale confirmation.
- **Trace:** [usage: recovery](../usage.md#clear-or-recover-a-gallery), [bin owner](../design-notes.md#recycle-bin-why-the-per-image-confirm-went); picked permanent deletion (`GAL#1`, 2026-10-18) and age-scoped Empty recycle bin (`GAL#5`, 2026-10-19) are built, recorded in the same bin owner section. There is no automatic purge.
- **Dependency:** UX#11 and UX#15; keep scale claims within `GAL#3`'s completed, bounded CPU checks and record new limits separately.

## Cross-cutting use

### UX#19 — Complete workstation tasks at realistic window sizes · P1 · Proposed

- **Basis:** evidence: responsive layouts exist; a running batch's progress fits 1536×864 at scroll 0 since 2026-10-20 (`UI#8`, a short-window panel tier); a maximized browser at that scaling (about 1536×758) still needs a scroll or the topbar's mirror. New full-cycle layouts remain hypotheses.
- **Story:** I can generate, judge, edit and recover work when my desktop window is wide, narrow or scaled.
- **Acceptance:** workbench trials cover existing smoke dimensions plus realistic short/scaled windows; critical status, action targets, dialogs and focus remain reachable without clipped content.
- **Acceptance:** layout changes do not move controls under the pointer, introduce panel scrolling or bypass existing under-1800px rules; any conflict is recorded for a deliberate decision.
- **Trace:** [layout owner](../design-notes.md#the-panels-sampling-and-output-rows), [tokens](../design-notes.md#design-tokens), `UI#8` (done 2026-10-20, the panel alone, under 900 px tall).
- **Dependency:** select design options first; retain current tokens and component ownership during any later implementation.

### UX#20 — Complete the same cycle with keyboard and assistive technology · P0 · Proposed

- **Basis:** L6–L8 Often confirms keyboard, image right-click and Ctrl/Shift selection as primary interaction paths; L1–L5 terminal tasks are Never and need no common-task layout space. Semantic/focus support exists; browser keyboard/forced-colour/reduced-motion coverage is `UI#9`, while hands-on screen-reader/touch validation remains `UI#7`. These checks are not self-reported feature use.
- **Story:** I can navigate, judge, select, edit and recover work without relying on a mouse, colour alone or visual status alone.
- **Acceptance:** validate complete tasks with an actual screen reader and keyboard; held-action reasons, status updates, modal focus/return and mask controls are understandable in use.
- **Acceptance:** forced colours, reduced motion and touch are exercised; sheet navigation is evaluated against `UI#4`, and arbitrary feature shortcuts are not presumed necessary.
- **Trace:** [usage: keyboard](../usage.md#keyboard-and-lightbox), [landmarks owner](../design-notes.md#headings-and-landmarks), [review limits](../review-2026-10-12.md#review--2026-10-12), `UI#4`, `UI#5`, `UI#7`, `UI#9`.
- **Dependency:** [validation](./validation.md) records observed outcomes; accessibility-tree inspection and fake-server success alone do not establish this outcome.

## Candidate delivery stages and dependencies

These stages organize design work and possible later delivery; they are not a
replacement backlog, approval gate or instruction to start implementation.

1. **Establish the workstation baseline:** use the recorded feature priorities,
   map current states to UX#1–UX#12, UX#14–UX#18 and UX#21, and record evidence gaps.
   Existing behavior is a constraint to understand, not a feature list to rebuild.
2. **Resolve continuity and scope:** preserve completed `GEN#11` under UX#10,
   evaluate UX#1 and UX#11 wording and handoffs, then test the
   immediate-iteration and morning-return journeys together.
3. **Improve judgment and curation:** evaluate direct inspection, sheet/Fill
   width, search and picked actions under UX#9 and UX#12 first. UX#13's
   arbitrary comparison and `UI#1`, `UI#3`, `UI#4` and `UI#6` are built;
   evaluate their placement rather than reopening their implementation.
4. **Close retention and recovery loops:** evaluate UX#15–UX#18 from keeper
   through archive/import and deletion recovery, preserving completed `GAL#2`
   and the bounded scale/interruption evidence for `GAL#3`.
5. **Validate the full cycle:** apply UX#19 and UX#20 throughout the trials,
   then repeat the selected complete journeys at representative sizes and
   input modes. Hardware quality/performance measurements remain separate.

Cross-stage dependencies are model/capability truth, reversible prompt edits,
explicit target scope, stored reproduction inputs and observable job state.
The workbench should show these together before individual conveniences are
ranked. Research, prototype review, CPU/browser checks and card measurements
answer different questions; validation must retain that distinction.

Deferred work stays deferred: external reference import (`GEN#8`), Widen/Load into panel,
advanced mask cropping (`INP#9`), later outpaint fills (`INP#10`)
and automatic image scoring (`PA#9`) require
their recorded conditions or a new explicit decision. No new TODO rows are created by this catalogue.
