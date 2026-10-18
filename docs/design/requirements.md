# Candidate design requirements

2026-10-18. Scope: Matt's personal Local Diffusion workstation on the RTX 5080,
including its Windows and WSL environments. These requirements describe his
whole working cycle; they do not propose a public service, accounts or a new audience.

Read [research](./research.md), [workbench](./workbench.md) and
[validation](./validation.md) alongside this catalogue. The
[usage guide](../usage.md) describes today's app; existing plans and
[TODO](../../TODO.md) retain ownership of implementation and its ordering.

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
budgets; no task frequencies or measured time savings are claimed.

## Workstation preparation

### UX#1 — Find the next useful action · P1 · Proposed

- **Basis:** hypothesis: the full cycle benefits from clearer orientation; current generation, results and gallery already share one page.
- **Story:** when I return to my workstation, I can tell what is ready, what is running and where to resume judging or creating.
- **Acceptance:** the workbench concept identifies the running checkpoint, active/paused/queued work and selected-image scope without opening unrelated dialogs.
- **Acceptance:** from cold, busy and returning-session fixtures, Matt can name the next action for his task before pressing it; record confusion rather than assuming a new navigation scheme is needed.
- **Trace:** [usage: jobs](../usage.md#generate-and-manage-jobs), [existing modes](../plan-continue-run.md#personas); no existing TODO item claims this full-app orientation outcome.
- **Dependency:** research's task priorities; preserve the existing no-build-step workstation architecture.

### UX#2 — Know which model is selected and usable · P0 · Current

- **Basis:** evidence: Models distinguishes selected, loaded, loading and failed states, saved choices and unsaved changes.
- **Story:** I can prepare the checkpoint I intend to use and understand what Apply & save or Switch & load will do.
- **Acceptance:** cold start, save without load, successful load, failed load and another-tab switch each show the effective checkpoint/options and an available next action.
- **Acceptance:** checkpoint changes update supported controls and defaults; a stale dialog or submission cannot silently use the previous model's assumptions.
- **Trace:** [usage: models](../usage.md#load-and-switch-models), [switching owner](../design-notes.md#switching-checkpoints-while-the-server-stays-open); related `GEN#1`, not closed by this baseline.
- **Dependency:** current model revision and load lifecycle rules remain authoritative.

### UX#3 — Judge preparation cost before committing time · P1 · Proposed

- **Basis:** evidence: first-use downloads and reference timings are documented; cache/download facts are open `GEN#1`. Whether they deserve more prominence is a hypothesis.
- **Story:** I can distinguish a ready local choice from one that may need downloads, approval or unmeasured resource use.
- **Acceptance:** a candidate model row separates cached/download information where known, access requirements, recorded speed references and unmeasured combinations; unknown values remain explicit.
- **Acceptance:** a reference timing is never presented as an ETA for my prompt or as proof that a size fits the card.
- **Trace:** [usage: models](../usage.md#load-and-switch-models), [verified measurements](../verified.md), `GEN#1`, `CARD#4`–`CARD#6`; live access preflight `GEN#7` stays deferred.
- **Dependency:** use existing sourced facts and local cache inspection; no model download or benchmark is required to explore the design.

## Prompting and immediate iteration

### UX#4 — Understand prompt advice and its limits · P0 · Current

- **Basis:** evidence: authored guides, token counts, heuristic findings, tag lookup and critic suggestions have different provenance and confidence.
- **Story:** I can decide which advice to use without mistaking a suggestion for a rule or an image-quality guarantee.
- **Acceptance:** documented guidance retains its source; heuristics and unchecked/unavailable results are identified; the critic's rewrite remains reviewable before adoption.
- **Acceptance:** inactive negative guidance, literal weighting syntax, network tag lookup and cold Check behavior are understandable in their relevant controls.
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [analyser owner](../design-notes.md#the-prompt-analyser); no new TODO row; `PA#9` automatic image scoring stays deferred.
- **Dependency:** checkpoint-specific guidance and current validation boundaries.

### UX#5 — Recover my text and settings after replacements · P0 · Current

- **Basis:** evidence: Reuse, PNG settings, examples, rewrites and expansion can replace working text; existing undo behavior protects it.
- **Story:** I can try advice or reuse an image without losing the prompt and controls I was editing.
- **Acceptance:** each replacement uses the established text undo path; Reuse/PNG restoration offers the documented settings Undo and names replaced prompt text and omitted inputs.
- **Acceptance:** edits made while Expand is waiting survive its answer; stale advice cannot silently overwrite newer text.
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [PNG settings](../usage.md#load-settings-from-a-png), [Reuse owner](../design-notes.md#per-control-defaults-and-reuse); related cancellation gap `GEN#2` remains open.
- **Dependency:** preserve separate prompt-text undo and whole-settings restoration semantics.

### UX#6 — Iterate from the result I am judging · P0 · Current

- **Basis:** evidence: the batch-iteration case motivated More seeds; continuation phase 1 is built on the finish line, image menu and lightbox.
- **Story:** when a result is promising, I can make more of its batch or reuse its settings while still judging it.
- **Acceptance:** batch and run entry points identify their target, images per cell, total work and seed policy before Queue; repeated submissions skip already planned seeds.
- **Acceptance:** lightbox/dialog Escape and return focus preserve the inspected image; narrow-width access remains available through the existing image menu.
- **Trace:** [usage: continue](../usage.md#continue-a-run), [accepted continuation design](../plan-continue-run.md#the-recommended-design); phase 1 is complete, `GEN#11` covers the separate restart gap.
- **Dependency:** preserve batch settings and joining the original run; a new continuation shortcut is not required.

### UX#7 — Understand work while it runs or waits · P0 · Current

- **Basis:** evidence: loading, helper work, queue waiting, pause, cancellation and terminal outcomes have distinct behavior.
- **Story:** I can tell what the workstation is doing and stop or hold the intended work without discarding saved results.
- **Acceptance:** representative loading, queued, paused, canceling, failed and completed states identify the work and relevant available action; a held action explains its reason when pressed.
- **Acceptance:** closing/reopening a tab reconnects to surviving server work; a server restart does not imply that queued jobs survived.
- **Trace:** [usage: jobs](../usage.md#generate-and-manage-jobs), [queue owner](../design-notes.md#the-browsers-queue); `UI#2` notifications and `GEN#2` Expand cancellation remain separate proposals.
- **Dependency:** single-worker lifecycle and cooperative cancel/pause contracts; no background auto-queueing.

## Experiments, comparison and continuation

### UX#8 — Predict a grid before running it · P0 · Current

- **Basis:** evidence: the grid preview resolves axes, prompt variables, existing cells and measured estimates before submission.
- **Story:** I can plan a bounded experiment and understand how many images it will create before using the card.
- **Acceptance:** a fixture crossing settings, seed count/list and prompt variables shows resolved prompts, total images, existing/queued cells and invalid or duplicate effective settings.
- **Acceptance:** budget wording describes starting no further cell, and unavailable estimates remain unavailable rather than inferred from unrelated measurements.
- **Trace:** [usage: grid](../usage.md#sweep-settings-with-grid-search), [preview owner](../design-notes.md#grid-preview-and-the-per-step-estimate); no new TODO row.
- **Dependency:** server expansion and normalization stay authoritative; existing image and batch guards remain intact.

### UX#9 — Compare a controlled experiment intelligibly · P0 · Current

- **Basis:** evidence: same-seed prompt comparisons and contact-sheet axes expose controlled differences; unrelated seeds do not isolate a prompt edit.
- **Story:** I can judge the effect of a change and understand what differed between the images.
- **Acceptance:** prompt versions remain identified within their submission, shared settings and changing axes are visible, and source/refine comparisons retain a meaningful crop.
- **Acceptance:** sheets and exported labels describe their scope; a small seed sample or a rewrite changing multiple things is not presented as conclusive evidence.
- **Trace:** [usage: prompts](../usage.md#write-and-check-prompts), [usage: gallery](../usage.md#browse-compare-and-keep-results), [sheet owner](../design-notes.md#the-contact-sheet); related `UI#3`, `UI#4` remain open.
- **Dependency:** distinguish same-seed experimental comparison from arbitrary aesthetic preference.

### UX#10 — Finish interrupted work after a restart · P0 · Accepted-unbuilt

- **Basis:** decision: continuation phase 2 records resolved job cells and supplies Finish run; today's Resume still depends on retained panel grid/page state.
- **Story:** the morning after a restart, I can finish a stored run without reconstructing its axes or base seed.
- **Acceptance:** Finish is held while that run has queued work, naming the job that holds the rest. Otherwise it previews and queues only missing, unplanned cells from stored jobs under the original run identity; binned cells are not redrawn.
- **Acceptance:** legacy/imported runs, changed model/quantizer, missing sources/adapters/masks, deleted results and storage failures follow the accepted phase-2 refusals and disclosures.
- **Trace:** [phase-2 owner](../plan-continue-run.md#phase-2-runs-keep-their-cells), [accepted answers](../plan-continue-run.md#open-questions-for-the-user), `GEN#11`; retain it as the next accepted implementation item after design validation.
- **Dependency:** persisted job definitions/outcomes and ordinary submission checks; this document does not authorize implementing phase 2.

### UX#11 — Predict the scope of every image action · P0 · Current

- **Basis:** evidence: picks, filter, whole run, all-checkpoint Starred and whole gallery are deliberately different scopes.
- **Story:** before I act, I know which images are included, including those hidden by the current view.
- **Acceptance:** Re-render, Swap tags, Mass refine and footer More seeds identify picks versus filter; run continuation identifies the whole run and ignores Starred as documented.
- **Acceptance:** bundle scope explains its all-checkpoint behavior; previews report counts, exclusions and resulting images without silently changing the target.
- **Trace:** [usage: selection/actions](../usage.md#browse-compare-and-keep-results), [bundles](../usage.md#gallery-bundles), [selection owner](../design-notes.md#the-gallery-selection); no new TODO row.
- **Dependency:** retain deliberate scope semantics; a consistent presentation pattern is evaluated in the workbench, not an assumed backend unification.

### UX#12 — Curate a set across searches without losing it · P1 · Proposed

- **Basis:** evidence: persistent picks and hidden-pick counts exist; Ctrl+A/Pick this run are open `UI#6`. Their usefulness and prominence require evaluation.
- **Story:** I can gather keepers across my runs, inspect them and apply an action to exactly that set.
- **Acceptance:** design trials retain picks and anchor across search/paging and a bin visit; hidden-pick counts remain understandable and Clear is deliberate.
- **Acceptance:** any proposed pick-all/run control states its loaded/run scope and cap before action; it does not quietly select an unlimited gallery.
- **Trace:** [usage: gallery](../usage.md#browse-compare-and-keep-results), [selection owner](../design-notes.md#the-gallery-selection), `UI#6`; a sheet of picks is not part of this requirement.
- **Dependency:** UX#11; keep current per-card star behavior and the separate temporary bin selection.

### UX#13 — Compare two chosen candidates · P1 · Proposed

- **Basis:** evidence: arbitrary Compare picked is open `UI#1`; existing comparison covers source/pass and CFG neighbors only.
- **Story:** when I have two candidates, I can inspect their visual and recorded-setting differences directly.
- **Acceptance:** the candidate design opens exactly two picked images, labels the pair and differing settings, and supports the existing fit/zoom and comparison modes where geometrically appropriate.
- **Acceptance:** missing/binned inputs and an unsupported picked count have explicit recoverable outcomes; closing comparison returns to the retained selection and focus.
- **Trace:** [usage: lightbox](../usage.md#keyboard-and-lightbox), [comparison owner](../design-notes.md#the-lightbox-zooms), `UI#1`.
- **Dependency:** UX#9, UX#11 and UX#12; evaluate the interaction before choosing its implementation or priority in TODO.

## Editing, retention and recovery

### UX#14 — Choose the edit that preserves what I intend · P0 · Current

- **Basis:** evidence: Refine redraws the source, Inpaint redraws a painted region, and Outpaint extends the canvas on enabled checkpoints.
- **Story:** I can improve a keeper while understanding which parts or composition may change.
- **Acceptance:** representative edit dialogs identify their source, output size, strength, versions, supported checkpoint and relevant measured/unmeasured guidance before running.
- **Acceptance:** Inpaint previews its painted mask and Outpaint its padding/band; held capabilities explain why and preserve keyboard access to that explanation.
- **Trace:** [Refine](../usage.md#refine-an-existing-image), [Inpaint](../usage.md#inpaint-part-of-an-image), [Outpaint](../usage.md#outpaint-an-image), [edit plans](../plan-inpainting.md), [outpaint plan](../plan-outpainting.md); `INP#3`–`INP#10` retain their own status.
- **Dependency:** current size guards and measured families; personal-file references `GEN#8` remain deferred until Matt raises them.

### UX#15 — Retain an edit's lineage and reproduce its intent · P0 · Current

- **Basis:** evidence: saved images retain sources, masks and outpaint specs; source removal affects reproducibility; Mass refine intentionally drops an inpaint mask.
- **Story:** I can trace a result back to its inputs and recognize when a rerun can no longer reproduce the original operation.
- **Acceptance:** facts, Reuse and rerun previews identify source and mask/spec; a binned source offers the documented recovery and a permanently missing one is not silently replaced.
- **Acceptance:** inpaint reruns preserve source size/mask, outpaint reruns preserve padding/fill/band, and an action that drops a mask states that resulting operation.
- **Trace:** [usage: edits](../usage.md#inpaint-part-of-an-image), [inpaint reproduction](../plan-inpainting.md#the-reproduction-set), [outpaint lineage](../plan-outpainting.md#lineage-and-gallery-actions); related measurement rows `CARD#1`–`CARD#3` remain open.
- **Dependency:** UX#14 and stored reproduction invariants; browser checks cannot establish pixel quality or GPU fit.

### UX#16 — Keep or share exactly the intended archive · P0 · Current

- **Basis:** evidence: backup, bundle-and-clear, picked/starred bundles and encrypted bundles already exist; chosen bundles intentionally omit unchosen dependency images.
- **Story:** I can retain my work or give away selected images with clear knowledge of what travels and what remains.
- **Acceptance:** whole-gallery backup, subset backup and bundle-and-clear distinguish retained/deleted contents, scope, omitted links and encryption behavior before action.
- **Acceptance:** only verified export permits clear; failures retain a reviewable result, and archive progress remains recoverable after closing the browser.
- **Trace:** [usage: bundles](../usage.md#gallery-bundles), [bundle owner](../design-notes.md#gallery-bundles-verify-before-clearing-stage-before-merging); `GAL#3` tracks scale/interruption qualification.
- **Dependency:** gallery maintenance admission and existing verification/staging; no automatic inclusion of unchosen images.

### UX#17 — Find and understand restored work · P1 · Proposed

- **Basis:** evidence: import preserves/remaps records and reports outcomes; Show imported is open `GAL#2`. A broader archive organizer is only a hypothesis.
- **Story:** after importing an archive, I can inspect the images just added and tell which dependencies were recovered or omitted.
- **Acceptance:** import results distinguish added, skipped and restored images, omitted links and failures; a Show imported route reaches the returned imported-image scope without manual id reconstruction.
- **Acceptance:** importing duplicate content preserves the documented current-gallery star/bin state; an empty import provides an intelligible result.
- **Trace:** [usage: bundles](../usage.md#gallery-bundles), [bundle owner](../design-notes.md#gallery-bundles-verify-before-clearing-stage-before-merging), `GAL#2`, `GAL#3`.
- **Dependency:** server import identity/range answer and UX#11; permanent archive browsing or project management needs separate evidence.

### UX#18 — Recover deletion and understand permanent loss · P0 · Current

- **Basis:** evidence: immediate binning with Restore replaces per-image confirmation; Empty bin and permanent delete disclose irreversible removal and affected sources.
- **Story:** I can discard rejects quickly, recover mistakes and recognize which permanent deletion would prevent later reproduction.
- **Acceptance:** single/bulk binning offers Restore, bin search stays within the bin, and visiting it preserves gallery picks; source dependencies remain visible.
- **Acceptance:** permanent actions confirm their actual scope and lineage consequence; changed contents cannot be removed under a stale confirmation.
- **Trace:** [usage: recovery](../usage.md#clear-or-recover-a-gallery), [bin owner](../design-notes.md#recycle-bin-why-the-per-image-confirm-went); `GAL#1` picked permanent deletion/age purge remains proposed, not included here.
- **Dependency:** UX#11 and UX#15; validate large bin operations under `GAL#3` before making scale claims.

## Cross-cutting use

### UX#19 — Complete workstation tasks at realistic window sizes · P1 · Proposed

- **Basis:** evidence: responsive layouts exist; short 1536×864 running-progress visibility is explicitly deferred `UI#8`. New full-cycle layouts remain hypotheses.
- **Story:** I can generate, judge, edit and recover work when my desktop window is wide, narrow or scaled.
- **Acceptance:** workbench trials cover existing smoke dimensions plus realistic short/scaled windows; critical status, action targets, dialogs and focus remain reachable without clipped content.
- **Acceptance:** layout changes do not move controls under the pointer, introduce panel scrolling or bypass existing under-1800px rules; any conflict is recorded for a deliberate decision.
- **Trace:** [layout owner](../design-notes.md#the-panels-sampling-and-output-rows), [tokens](../design-notes.md#design-tokens), `UI#8`; evaluation does not promote that deferred implementation.
- **Dependency:** select design options first; retain current tokens and component ownership during any later implementation.

### UX#20 — Complete the same cycle with keyboard and assistive technology · P0 · Proposed

- **Basis:** evidence: keyboard, semantic and focus support exist; real screen-reader, forced-colour, reduced-motion and touch use remain unverified `UI#7`.
- **Story:** I can navigate, judge, select, edit and recover work without relying on a mouse, colour alone or visual status alone.
- **Acceptance:** validate complete tasks with an actual screen reader and keyboard; held-action reasons, status updates, modal focus/return and mask controls are understandable in use.
- **Acceptance:** forced colours, reduced motion and touch are exercised; sheet navigation is evaluated against `UI#4`, and arbitrary feature shortcuts are not presumed necessary.
- **Trace:** [usage: keyboard](../usage.md#keyboard-and-lightbox), [landmarks owner](../design-notes.md#headings-and-landmarks), [review limits](../review-2026-10-12.md#review--2026-10-12), `UI#4`, `UI#5`, `UI#7`.
- **Dependency:** [validation](./validation.md) records observed outcomes; accessibility-tree inspection and fake-server success alone do not establish this outcome.

## Candidate delivery stages and dependencies

These stages organize design work and possible later delivery; they are not a
replacement backlog, approval gate or instruction to start implementation.

1. **Establish the workstation baseline:** confirm Matt's real task priorities,
   map current states to UX#1–UX#11 and UX#14–UX#18, and record evidence gaps.
   Existing behavior is a constraint to understand, not a feature list to rebuild.
2. **Resolve continuity and scope:** carry accepted `GEN#11` through UX#10's
   specification, evaluate UX#1 and UX#11 wording and handoffs, then test the
   immediate-iteration and morning-return journeys together.
3. **Improve judgment and curation:** compare candidate interactions for
   UX#9, UX#12 and UX#13 with Matt; consider `UI#1`, `UI#3`, `UI#4` and
   `UI#6` only where task evidence supports them.
4. **Close retention and recovery loops:** evaluate UX#15–UX#18 from keeper
   through archive/import and deletion recovery; `GAL#2` and `GAL#3` retain
   their owners and distinct implementation/qualification work.
5. **Validate the full cycle:** apply UX#19 and UX#20 throughout the trials,
   then repeat the selected complete journeys at representative sizes and
   input modes. Hardware quality/performance measurements remain separate.

Cross-stage dependencies are model/capability truth, reversible prompt edits,
explicit target scope, stored reproduction inputs and observable job state.
The workbench should show these together before individual conveniences are
ranked. Research, prototype review, CPU/browser checks and card measurements
answer different questions; validation must retain that distinction.

Deferred work stays deferred: external reference import (`GEN#8`), Widen/Load into panel,
advanced mask cropping (`INP#9`), later outpaint fills (`INP#10`),
automatic image scoring (`PA#9`) and short-window tightening (`UI#8`) require
their recorded conditions or a new explicit decision. No new TODO rows are created by this catalogue.
