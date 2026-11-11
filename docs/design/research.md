# Design discovery: the personal workstation

**Date:** 2026-10-18. **Status:** repository evidence and proposed design research;
no new task observation, interview, usability trial or GPU measurement completed.
**Owner:** Matt. **Scope:** AuraDream as Matt's personal image workstation.

This discovery broadens the recent Continue a run design into the whole working
cycle: start, choose a model, author, generate, experiment, recover, judge, edit
and archive. The user's session direction is **personal workstation only**.
The three personas below are modes of Matt's work, not separate audiences.

**Owner priority, this session:** "i want to be able to do common tasks
quickly." Treat speed through familiar tasks as the leading design objective.
The subsequent [questionnaire responses](preferences.md) supply all 129
reported feature frequencies and one area rating. Measured task baselines
remain absent. The first A1–A14 answers and the separate Reset decision are
preserved in the fuller responses.

Use the [design index](README.md) for the programme,
[workbench](workbench.md) for the proposed experience,
[requirements](requirements.md) for the requirements derived from this evidence,
and [validation](validation.md) for the personal task checks still to perform.
This file records discovery; it does not replace implementation rules, the
current feature plans, or measurements in `docs/verified.md`.

**Status reconciliation, 2026-11-03:** the journey and persona `Current`
entries below, F02, F04, E08 and E14 now describe today's app: both
continuation phases, model cache facts, Expand Stop, named setups, saved
searches, Then refine, the masked-area crop and Edit by instruction are
built, and Matt rated those four Sometimes ([later features](preferences.md#later-features-2026-11-03)). Goals, needs and non-goals are unchanged, and the register keeps
its 2026-10-18 line references; no new task observation was made. Questions 3, 6 and 7 are partly answered.

## Scope and evidence limits

- The product is for Matt's own workstation. No invented beginner, customer,
  team, public service, account system or commercial audience is included.
- The existing mode model comes from recorded decisions and repository review,
  not interviews. It supports useful questions, not claims about task frequency.
- The 129 answers are Matt's self-reported feature use, recorded exactly in
  preferences.md. They do not supply measured timings or create unreported
  area ratings, and are not an observed usability trial.
- Previous code reviews identify affordances and failure paths; they do not
  establish whether Matt finds a task easy or how he actually performs it.
- The Continue a run plan's empty-gallery and bundle observations are dated
  evidence from that plan. This discovery does not recheck private data.
- No live gallery, protected bundle, personal image or saved prompt was read.
- Current behaviour below is supported by the checked repository. Target
  journeys are design intentions to evaluate, not descriptions of shipped UI.
- CPU and browser checks cannot establish image quality, card fit or timing.
  GPU work remains subject to Matt's approval and the measured card limits.
- Both phases of Continue a run are built. Its old persona frustration
  paragraphs describe an earlier state and must not be reported as current
  defects.
- The [2026-11-01 first-run review](../maintainer/review-2026-11-01-first-run.md)
  (installer, doctor, first-visit panel, plain-words failures, `GEN#41`) was
  requested by Matt for a first-time user who is not confident with git, and
  serves the public repository. It sits outside these modes and is owned by
  that review. On 2026-11-03 Matt decided it gets no persona or mode of
  its own (first run or reinstall): it stays outside the design programme.

## Evidence register

Line references describe the checkout inspected on 2026-10-18; section links
remain useful as files grow. `D` is direct session direction, `R` recorded
owner decisions or requirements, `B` documented/source behaviour, `M` measured
evidence, and `H` a hypothesis that still needs a personal task check.

| ID | Kind | Evidence and exact repository source | What it supports |
|---|---|---|---|
| E01 | D | This session, 2026-10-18: full-scale design; personal workstation only; "i want to be able to do common tasks quickly" | Whole-cycle scope; one owner; common-task speed leads |
| E02 | R | [Continue a run: evidence and personas](../maintainer/plan-continue-run.md#evidence-and-its-limits), lines 47-126 | One user; three modes; no interview or usage-frequency evidence |
| E03 | R | [Accepted continuation answers](../maintainer/plan-continue-run.md#open-questions-for-the-user), lines 652-692 | Batch-based count; preview dialog; joining runs; phase-2 Finish; deferred widening |
| E04 | B | [Load and switch](../usage.md#load-and-switch-models), lines 9-124; [startup offer](../../web/model-switch.js#L741), lines 741-747 | Cold start, model choice, load costs, persistence and access failure |
| E05 | B | [Generate and manage jobs](../usage.md#generate-and-manage-jobs), lines 128-154 | Batch/Repeat, queue, pause/cancel, reload and per-model settings |
| E06 | B | [Write and check prompts](../usage.md#write-and-check-prompts), lines 184-249 | Examples, tags, Check, critic, comparisons and Expand |
| E07 | B | [Grid search](../usage.md#sweep-settings-with-grid-search), lines 367-413 | Resolved previews, seeds, axes, budgets and current Resume limits |
| E08 | B | [Current continuation](../usage.md#continue-a-run), lines 522-556; [finish actions](../../web/job-results.js#L121), lines 121-214 | As of 2026-11-03: More seeds, and Finish run from stored cells after a reload or restart (both phases, `GEN#11`); the panel's Resume tick only for older/imported runs |
| E09 | B | [Browse and keep](../usage.md#browse-compare-and-keep-results), lines 417-518 | Model scope, run/job scope, sheets, picks and batch actions |
| E10 | B | [Editing](../usage.md#refine-an-existing-image), lines 270-363 | Separate edit dialogs, source links, masks, padding and capability limits |
| E11 | B | [Bundles](../usage.md#gallery-bundles), lines 651-732; [recycle bin](../usage.md#clear-or-recover-a-gallery), lines 736-759 | Export scope, verified clearing, import and reversible triage |
| E12 | B | [Panel persistence](../design-notes.md#panel-persistence), lines 4533-4577; [Expand text ownership](../../web/prompt-expand.js#L146), lines 146-152 | Remembered work, Undo and preserving intervening edits |
| E13 | B | [Rules](../rules.md#rules), lines 1632-1686, 2024-2037, 2124-2134, 3061-3066 | Cold sourced checking, browser verification, discoverable holds, stable layout |
| E14 | B | [NOW and open work](../../TODO.md#now), lines 11-16, 56-63, 90-97 | As of 2026-11-03: durable recovery (`GEN#11`) and the model cache/Expand gaps (`GEN#1`, `GEN#2`) are closed; open work is `CARD` rows and what needs Matt (`DEV#1` trials, `UI#7` hands-on checks) |
| E15 | M | [Measured evidence](../verified.md); [measurement policy](../../TODO.md#measurements-on-the-card), lines 16-20 | Card evidence is distinct from design or fake-server checks |
| E16 | D | [First partial generation-control answers, 2026-10-18](preferences.md) | Historical first response: A1–A14 only; primary prompt/negative, aspect/resolution, fixed Seed, Batch/Repeat and Reset; secondary sampling |
| E17 | D | [Earlier partial pasted response and Reset decision, 2026-10-18](preferences.md) | Historical intermediate response: 115 feature answers before B arrived; Grid area Sometimes; Reset still clears Seed |
| E18 | D | [Latest complete pasted response, 2026-10-18](preferences.md) | 129 feature answers: 66 Often, 34 Sometimes, 29 Never; B adds frequent Examples/guide/autocomplete/Tags/chips/Check; prior ratings and Reset retained |

## Mode personas

These preserve the existing modes in E02. Goals are grounded in that work;
needs below translate them into design concerns. No biography, demographic,
skill rating or unobserved preference is assigned to Matt.

### P1. Queued Experimenter (earlier Night Runner)

- **Trigger:** queue work while another job runs, return to ongoing work,
  or occasionally configure a multi-setting experiment.
- **Goal:** see progress and the intended job, cancel or retry the correct
  work, retain results and continue without reconstructing the plan.
- **Current means:** queue/watch/cancel/reconnect/retry, recent results,
  prompt variables, occasional Steps/CFG/seed grids, runs and sheets
  (E05, E07, E08, E17). Since then: Finish run, Skip and Move to front,
  Then refine and queue estimates. The page watches the running job;
  choosing the watched job is not built and is Proposed (see UX#7's 2026-11-03 note in
  [requirements](requirements.md#ux7--understand-work-while-it-runs-or-waits--p0--current)).
- **Needs:** persistent progress and explicit waiting/outcome; quick watch,
  cancel and retry; saved results retained; a trustworthy recovery path.
- **Decision point:** which job to inspect, cancel, finish or extend, and
  whether the occasional comparison needs another setting or seed.
- **Non-goals:** automatic continuation, unlimited overnight work, speculative
  hardware promises, or reopening the deferred Widen decision (E03, E15).
- **Reported refinement:** time budgets and Pause are Never, while the
  queue/cancel/retry/reconnect paths are Often. The earlier Night Runner's
  overnight/budget story remains historical context, not the priority for
  this layout. Exact session sequences and timings remain unknown.

### P2. Batch Iterator

- **Trigger:** an idea to try, a prompt adjustment, or a promising small batch.
- **Goal:** judge quickly, change an intentional input, and obtain more useful
  variations while retaining the settings that made the batch promising.
- **Current means:** model defaults, remembered controls, Generate, Use seed,
  Reuse, Check, prompt comparison and More seeds (E03-E08). Since then:
  named setups (Restore/Undo), personal snippets, random prompt choices
  and the tag finder.
- **Needs:** quick access to the prompt and latest result; visible active
  settings; reversible text changes; explicit scope and count before queuing.
- **Decision point:** preserve a seed for comparison, vary the seed, change the
  prompt/settings, or move into an edit that starts from the image's pixels.
- **Non-goals:** every generation requiring analysis, advice silently replacing
  the prompt, or reducing reproducibility to the seed alone.
- **Reported control priorities:** prompt/negative, aspect/resolution, fixed
  Seed, Batch/Repeat and Reset are frequent; Steps/CFG/Mu are occasional
  (E16). Reuse, Use seed, all More seeds scopes, Re-render and Swap tags are
  frequent (E17). The order of changes between Generate presses is unknown.

### P3. Curator

- **Trigger:** a set is ready to judge, a keeper needs work, or a session ends.
- **Goal:** identify worthwhile images, compare alternatives, improve selected
  results and retain a recoverable archive with useful provenance.
- **Current means:** run/job filters, sheet/lightbox, stars, persistent picks,
  Refine/Inpaint/Outpaint, recycle bin and bundles (E09-E11). Since then:
  saved searches, Compare picked, Pick this run, Only the masked area and
  Edit by instruction.
- **Needs:** a reliable comparison context; visible action scope; clear source
  relationships; reversible triage; a verified archive result before clearing.
- **Decision point:** keep, request more variations, edit, bin, or archive.
- **Non-goals:** a separate asset-management product, automatic quality scoring,
  an automatic purge, or new sharing/account workflows.
- **Reported refinement:** gallery search/star/all-model scope and picking,
  full-size facts/zoom, source comparison, sheets/Fill width, Refine and
  Mass refine, and whole-gallery password archives are frequent. Gallery
  beside newest, sheet export and picked/starred archives are Never (E17).
  The qualities judged first and session-completion cue remain unknown.

## Jobs to be done

| ID | When | I want to | So I can | Evidence |
|---|---|---|---|---|
| J01 | Opening the workstation | Recognise the server, model and remembered state | Continue with an informed first action | E04, E05, E12 |
| J02 | Choosing a checkpoint | Judge creative suitability and the cost of using it | Commit to a model that suits this attempt | E04, E14, E15 |
| J03 | Writing a prompt | See relevant examples, limits and optional grounded advice | Make my own deliberate prompt choices | E06, E13 |
| J04 | Trying an image idea | Queue an explicit amount of work with known active inputs | Judge results without losing my draft | E05, E12 |
| J05 | Comparing a change | Hold the relevant inputs constant and view alternatives together | Decide what the change actually improved | E06, E07, E09 |
| J06 | Returning after a stop | Distinguish missing work from more variations | Recover or extend the intended set | E03, E07, E08, E14 |
| J07 | Reviewing results | Compare, keep and choose an action at the image or set | Move from exploration to a useful result | E08-E10 |
| J08 | Improving a keeper | Choose whole-image refinement, a painted region (whole picture or only the masked area), extra canvas or a written instruction (Edit by instruction, since 2026-10-30) | Change the intended part with clear provenance | E10 |
| J09 | Ending a session | Verify what is archived and what will leave the gallery | Clear space while retaining work I mean to keep | E11 |

## Current and target journeys

Targets describe desired outcomes. The workbench and requirements turn them
into candidate structure; the validation plan checks them with Matt.

| Stage | Current journey | Target journey to evaluate | Evidence |
|---|---|---|---|
| Start | Launch from the shortcut or `auradream --open`; a second launch says the server is already running; restore panel/settings; idle cold first connection offers Models without taking the caret, its first-visit note saying closing the panel saves nothing; without a Hugging Face token it preselects an open checkpoint (`GEN#41`) | Immediately recognise what is restored, whether the server answers, what is loaded and the useful next action | E04, E05, E12 |
| Model | Inspect catalogue facts and own examples; choose loader settings; Apply & save or Switch & load; recover a failed load | Choose with creative fit, access and preparation cost understood; see selection versus loaded state clearly | E04, E14 |
| Prompt | Write; optionally use Examples/Tags/Check; optionally ask critic or Expand; review/apply with Undo | Keep authorship central; disclose checkpoint-specific limits and optional help at the moment it is useful | E06, E12, E13 |
| Generate/experiment | Generate with Batch/Repeat, or configure grid/comparison; restore a named setup (with Undo); optionally Then refine; preview; queue; Skip or Move to front; watch results | Make a small attempt quickly, then make a controlled comparison without reconstructing the question | E05-E07, E09 |
| Recover | Browser reload reattaches; server restart ends jobs; saved images remain; Finish run checks stored missing cells after a reload or restart; the Resume tick only for older/imported runs | Recognise completed, missing and queued work; finish an interrupted run or request more seeds with explicit totals | E03, E07, E08, E14 |
| Curate | Narrow model/run/job/filter or apply a saved search; inspect sheet/lightbox; star, pick or Pick this run; Compare picked; selected images survive view changes | Preserve comparison context and understand exactly which image, picked set, filter or run an action will use | E08, E09 |
| Edit | Open Refine, Inpaint or Outpaint on a source, optionally Only the masked area; or Edit by instruction (FLUX.1 Kontext [dev], edit-only, inside its measured envelope; card acceptance `INP#11` pending); set prompt/strength/versions; inspect output and source comparison | Choose the operation by intended change, retain the source and recognise capability or reproduction limits | E10 |
| Archive | Choose whole/picked/starred backup or whole-gallery Bundle & clear; read verified result; import to recover. A default-off Automatic startup backup section exists; Matt declined unattended backups 2026-10-29 ([plan](../maintainer/plan-automatic-backups.md#matts-choices-for-unattended-backups)) | Know archive scope and omitted dependencies before acting; finish with a comprehensible verified outcome | E11 |

The current More seeds entry points are assets to retain, not new proposals.
The recovery target incorporates the already accepted phase-2 direction; it
does not rename the panel's Resume tick or live-job Pause/Resume (E03).

## Facts, interpretations and hypotheses

| ID | Statement | Classification | What would resolve it |
|---|---|---|---|
| F01 | Check works cold; advice and rewrites are optional; text edits support Undo | Repository behaviour, E06/E12/E13 | Preserve through design and browser checks |
| F02 | Runs do not yet retain resolved cells for durable Finish run | Closed 2026-10-18: `GEN#11` (858f3a2, 3c8353a) | Accepted phase-2 implementation and restart check |
| F03 | More seeds is available at finish lines, run surfaces and image actions | Current phase-1 behaviour, E08 | Preserve locality and explicit scope |
| F04 | Model download/cache visibility and cancellation of Expand remain open | Closed 2026-10-18: `GEN#1` cache/download facts (5f1d592), `GEN#2` Expand Stop (0ba0c2e) | Requirements and implementation decisions |
| H01 | Choosing the right workflow is harder than operating its controls | Untested interpretation of overlapping operations | Ask Matt to choose a route from concrete intentions |
| H02 | A coherent workbench will reduce repeated navigation between authoring and judgement | Design hypothesis | Compare current and candidate task paths |
| H03 | Model facts need stronger decision support, beyond an accurate table | Design hypothesis | Observe a real model-choice decision and its missing information |
| H04 | Durable run recovery should be the organising centre of unattended work | Design hypothesis beyond accepted Finish | Test morning review and recovery without assuming a new top-level destination |
| H05 | Action scope is the main source of uncertainty in curation and batch work | Design hypothesis | Ask Matt to predict scope before each action and compare with the preview |
| H06 | Progressive disclosure can reduce clutter without hiding daily controls | Design hypothesis | Try ordinary iteration and less frequent edits at Matt's actual window sizes |

## Questions for Matt

These are discovery questions, not permission gates. They do not reopen accepted
continuation decisions or promote deferred features into the current scope.
The fuller questionnaire is sufficient for the next layout revision; no
additional answer is required now. Use these only when a specific upcoming
design decision or task trial needs them. Do not send the entire list again.

1. **Priority detail:** use the confirmed frequent tasks to choose a short
   sequence to time first; task order and detours are still unobserved.
2. **Success:** when judging a batch, what makes one image worth keeping, and
   which details do you inspect before deciding?
3. **Iteration:** after an almost-right result, what do you usually try next:
   prompt, fixed seed, more seeds, settings, Refine, Inpaint or another model?
   *Partly answered:* Matt doesn't run klein ([TODO](../../TODO.md) `INP#5`),
   doesn't write weighted prompts (`GEN#20`), and reopened `INP#9` (only the
   masked area, built 2026-11-03). The usual next step remains open.
4. **Experiment:** what recent question did you want a sweep to answer, and
   what comparison arrangement would make its answer clear?
5. **Return:** when you come back, what do you need to recognise first: last
   result, run progress, missing work, active model, or the draft prompt?
6. **Workspace:** which window size, display scaling and input methods should
   the primary layout be judged on, alongside the existing smoke sizes?
   *Partly answered:* Matt chose to fit 1536x864, 1080p at 125%
   ([design notes, `UI#8`](../design-notes.md#the-panels-sampling-and-output-rows),
   2026-10-20). Input methods remain open.
7. **Archive:** what tells you a session is ready to bundle, and what do you
   need to recognise when bringing an old bundle back?
   *Partly answered:* no unattended backups, 2026-10-29
   ([plan](../maintainer/plan-automatic-backups.md#matts-choices-for-unattended-backups)).
   The bundle and return cues remain open.

## Personal task validation

Matt is the participant and product owner; no external recruitment is planned.
Use agreed example prompts and scratch galleries for repeatable checks. A real
session may add evidence when Matt chooses it; record task outcomes without
copying private images, prompts or bundles into design documentation.

Validate a cold start/model decision, a small prompt iteration, a controlled
comparison, an interrupted-run return, keeper selection, a local image edit,
and archive/recovery. Before each action, ask what Matt expects it to affect;
afterwards, record completion, hesitation, wrong turns, preserved work and
whether the result matched that expectation. Keep fresh observations separate
from the repository baseline and give each finding a date and evidence link.

The [validation plan](validation.md) owns task scripts and acceptance evidence.
Passing functional smokes demonstrates operation; successful personal task
checks support usability claims. Neither establishes GPU quality or speed.
