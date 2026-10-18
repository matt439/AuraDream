# Local Diffusion: full-app design

**2026-10-18 · candidate design · personal workstation only.** This expands
the feature-level [Continue a run design](../plan-continue-run.md) into an
end-to-end design for Matt's workstation. It is a product and interaction
design baseline, with an executable concept for review. It does not mark
the proposed interface as built or usability-tested.

## Product brief

Local Diffusion should let Matt turn an idea into useful images, run bounded
experiments, judge the differences, improve keepers, and preserve the result
without losing the settings or context needed to continue later.

The unit of success is a useful creative or experimental outcome, rather
than the number of features exposed or images generated. The application
already has substantial functionality. This design primarily addresses how
that functionality fits together and how clearly it communicates its scope.

**Primary objective, confirmed this session:** "i want to be able to do
common tasks quickly." Optimize interaction time, detours and recovery of
context in the normal loop. Hardware speed is a separate measurement.
Matt has now supplied [all 129 feature ratings](preferences.md): 66 Often,
34 Sometimes and 29 Never, plus Grid experiments overall as Sometimes.
These are self-reports; task timings remain unobserved.

**Audience decision:** Matt answered **personal workstation only** on
2026-10-18. Personas remain modes of his work. General onboarding, accounts,
collaboration, commercial service design and hypothetical customer segments
are outside this brief. Accessibility and recovery still matter to the
reliability of this workstation.

## Read the design

| Artifact | What it answers |
|---|---|
| [Research and journeys](research.md) | What is known, what is inferred, whose needs drive the work, and where each journey begins and ends |
| [Reported feature use](preferences.md) | Matt's exact frequency answers and the resulting generation-control priorities |
| [Workbench design](workbench.md) | Information architecture, vocabulary, layout, interaction contracts, visual direction and complete states |
| [Requirements](requirements.md) | Outcome-based stories, acceptance criteria, existing feature owners and candidate delivery sequence |
| [Options and validation](validation.md) | Alternatives, trade-offs, task protocol, provisional targets and evidence needed before implementation |
| [Interactive concept](prototype/index.html) | A sample workbench for exploring views, action scopes, comparison, draft preservation and recovery |
| [Prototype checks](prototype/check.json) | Repeatable browser checks of the concept's working interactions |

The existing [rules](../rules.md), [design notes](../design-notes.md),
[usage guide](../usage.md), [measurements](../verified.md) and feature plans
retain their authority over today's app. This directory owns the proposed
whole-app design. [TODO](../../TODO.md) remains the one index of open work;
requirements here do not create a second implementation queue.

## Intended outcomes

1. **Stay in the creative loop.** Generate, inspect, change or extend, and
   inspect again with the draft and relevant results still available.
2. **Know what an action affects.** Image, batch, picked set, filter, run and
   whole-gallery actions state their scope and counts before commitment.
3. **Trust unattended work.** Distinguish waiting, progress, pause, budget
   stop, failure and completion; recover the unfinished plan without
   reconstructing it from memory.
4. **Judge changes fairly.** Keep source relationships, comparison context
   and differing settings visible enough to support a decision.
5. **Finish the lifecycle.** Keep, edit, back up, bundle, clear and recover
   images with deliberate inclusion of dependencies and readable outcomes.

## Candidate direction

Use **one workbench with a persistent draft, model/queue status and a shared
visual workspace**. The workspace can focus on **Results**, **Runs** or
**Gallery**. These are views, not required stages. The image inspector and
run context expose the next appropriate actions where Matt judges the work.

This is a reasoned design recommendation, not an approved replacement for
the current layout. Compare it against keeping today's arrangements in the
[task trials](validation.md). A separate project/notebook system is not
justified by the evidence currently available.

## Decisions and boundaries

| State | Decision |
|---|---|
| Confirmed this session | Personal workstation only |
| Confirmed use | Generation, queue control, searching/picking, full-size/sheet review, continuation, reruns/Refine and whole-gallery archiving are central; see the exact feature ratings |
| Existing accepted direction | More seeds keeps its preview/confirmation, retained batch semantics and original run; phase 2 adds persisted cells and Finish run |
| Existing safeguards | Undoable prompt writes; stable controls; no automatic page scrolling; explicit held-action reasons; authoritative server previews |
| Proposed here | Results/Runs/Gallery views with shared context; consistent scope presentation; inspector hierarchy; consolidated recovery and review journeys |
| Open research | Actual task sequences/timings; image space versus adjacent detail; whether view changes improve the normal loop |
| Still deferred | External photo/sketch references, Widen, Load into panel, and other explicitly deferred features in TODO |

The fuller answers make the next layout concrete: keep the primary composer,
make queue/watch/cancel/retry status direct, give Gallery and contact sheets
the main visual space, and expose scope-specific reuse/More seeds/rerun/Refine
actions. Whole-gallery password backup, clear, download and import belong
where curation ends. The existing Gallery-beside-newest mode, Pause, advanced
loader settings, manual conditioning and terminal tools need secondary
access. No feature is removed. Reset retains today's clear-seed behavior.

Prompt variables deserve a direct route despite grids overall being
occasional. The latest B answers also keep Examples, guidance, Tags,
autocomplete, chips and Check beside authoring. Critic/rewrite/comparison
and ERNIE Expand stay contextual, with PNG settings secondary.

## Prototype scope

Open [prototype/index.html](prototype/index.html) directly in a browser. It
uses synthetic geometric placeholders and in-memory sample records; no
image quality or usage statistics are implied. It calls no application
API, opens no gallery, stores no settings and runs no model.

Working paths include changing views without losing the draft, selecting
across filters, image/run scope previews, comparing two picks, and simulating
Finish run. The preference revision adds independent Aspect/Resolution,
Batch × Repeat counts, Reset/Undo, watched-job cancellation, a query-based
sample sheet/Fill width, and source comparison. Prompt helpers, frequent
scope actions and whole-gallery archive/model controls have visible routes.
Some routes show their scope and placement only; they do not emulate the
production planner or helper.

Finish run is explicitly labelled as an accepted future capability, and
arbitrary Compare picked remains proposed. Queueing records a simulated
outcome only. Refinement execution, painting, real model loading and archive
operations remain specified rather than implemented in the concept. Its
two independent job fixtures and sample run outcomes are not a durable
production queue. All sample values are illustrative, including checked
archive/model choices; no saved preference is inferred or changed.

## Delivery and validation status

- Repository discovery and source-based workflow audit: completed.
- Whole-app brief, journeys, requirements and candidate interaction design:
  written; no new audience or interview evidence invented.
- Clickable concept: provided for review; verification results are recorded
  in [validation.md](validation.md).
- Matt's task trials, everyday-session evaluation, assistive-technology
  checks and target-card measurements: not performed in this design pass.
- Application implementation: unchanged by this design work.

Start the next design session with the task set in validation.md and Matt's
actual frustrations. Revise the candidate from those findings, then select
implementation slices in TODO. Preserve the accepted GEN#11 owner plan.
