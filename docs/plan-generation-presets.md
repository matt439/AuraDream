# Named generation and grid setups

**Status: done 2026-10-28** (`GEN#27`).

`GEN#27`. Requested for the backlog on **2026-10-28**, after the TODO audit.
**Status: stages 1-3 built and locally qualified, 2026-10-28.**
This plan defines CPU/browser implementation slices; it does not
commission a model load, download or card trial.

## Purpose and boundary

Save several reusable setups instead of reconstructing one from an old image
or overwriting the browser's sole current draft. A setup belongs to the exact
checkpoint id and contains editable generation settings or an unexpanded grid
definition. It is neither a recorded image nor a retained run.

Existing [personal snippets](plan-prompt-snippets.md) remain named text pieces
that Append or Prepend. A setup replaces a defined set of composer fields with
Undo. Existing [panel persistence](design-notes.md#panel-persistence) still
remembers the current browser draft; opening or editing a saved setup does not
replace that draft until Restore is pressed. The
[reported use](design/preferences.md) supports prompt variables, fixed seeds,
Batch/Repeat and Reuse as useful targets; it does not establish a measured
workflow speed gain.

Version 1 covers ordinary text-to-image and grid composition. It does not save
LoRAs, gallery references, Refine/Inpaint/Outpaint/instruction-edit sources or
Then refine. Save is held when any such active input would be omitted, with
words naming it. Restore explicitly clears current conditioning and turns
Then refine off; its preview names these changes and Undo restores them.
There are no image, source, mask, job or run ids, completed-cell lists, Resume
flags, queue state, estimates or results in a setup. No gallery schema, PNG
chunk, CellKey or run identity changes are needed.

## Saved contract

Store a versioned collection in `auradream.presets.json` beside the database,
independent of gallery rows and bundles. Each item has a server-owned random
id, positive optimistic revision, readable name, exact checkpoint id and a
typed payload. Reject unknown fields and newer versions. Use the snippet
sidecar's bounded, cross-process locked read/modify/write and atomic
replace pattern; an unreadable file is an error, never an empty replacement.
Ignore the sidecar and temporary files in Git.

The payload contains:

- Exact Prompt and Negative text, width/height, and sampling overrides for
  Steps, CFG, Mu and True CFG. A null override means follow that checkpoint's
  current profile; explicit zero remains zero. Do not turn a profile default
  into a user-owned override by reading only the displayed number.
- A nullable base seed, Batch and Repeat. Null seed restores an empty Seed
  box; the request receives its actual seed at ordinary submission, never at
  Save or Restore. A fixed seed is restored exactly within the existing limit.
- A Generate/Grid mode. Grid stores the variable block and cross/random mode,
  Steps/CFG/Mu/Size axis strings, seed count or explicit seed list and optional
  time budget. Keep their text and value order; no substitution, draw or
  expansion is persisted. A Generate setup has no active grid definition.

Reuse existing request limits for every generation/grid field. Initially cap
the collection at 100 items, names at 80 characters and its serialized file at
8 MiB. Validate all items on read, and preserve the previous file on failed
writes. Setup names and payloads are private local data, rendered as text.

Do not serialize `panelState()` wholesale: its browser, per-model history,
translation, references and performance envelope have different ownership.
Version 1 has no loader or execution fields. It is available only with valid
Eager/native loader intent and step cache Off. Save and Restore refuse invalid
or held performance intent before changing anything; they never clear a held
choice, substitute a default, activate a recipe or switch/load a checkpoint.
An unexpected performance field in a file is a schema error. Later execution
support requires an explicit versioned contract and the existing availability
guards; a saved setup cannot grant it.

## Validation and restoration

CRUD is model-free and independent of the generation gate and gallery
maintenance. Use bounded collection/detail reads and revisioned create, update
and delete routes, following snippets. No startup read, Hub request, tokenizer,
model load or CUDA access is needed. Reading a setup for another checkpoint
remains possible. Saving a captured composer checks its checkpoint/revision so
a concurrent model switch cannot mislabel the payload.

Separate the editable payload from its effective request. The server validates
typed fields and grid grammar with the existing Generate/Grid schemas and
pure grid helpers. The existing `GenerateRequest.applied()` remains the one
normalization of effective cells, including negative activation and sampling.
Do not invent a second normalization, store applied text over the user's
draft, or use a saved setup as an alternate submission endpoint. Cold checks
must not retain work or assign a seed to the draft. A random-grid preview may
use a temporary sample under the existing preview policy, labelled as example
prompts; that sample is never adopted by Save or Restore.

Restore requires the exact selected checkpoint id. On a mismatch, hold Restore
with the checkpoint's name and offer the existing Models dialog; opening it
does not switch or load. Revalidate against the current profile, capabilities
and model revision before writing. Show saved versus effective values and any
inactive negative/sampling input; unsupported active grid axes or other
unusable inputs hold Restore rather than disappearing silently. Restore keeps
raw negative text even when inactive, as the composer already does.

Generate restores Batch and Repeat as saved. Grid uses the existing seed-axis
and Batch semantics, forces request Repeat to 1, and leaves Resume unticked.
Its ordinary server preview supplies counts, resolved prompts, limits and
estimates. Random variables retain their existing per-seed draw; restoring
never recovers an old run or promises identical future pixels. Neither mode
submits work until the user presses Generate or Run grid normally.

A late validation answer cannot replace a newer draft. Bind it to the item
revision, model revision and captured composer state; if any changes while
waiting, require a new deliberate Restore press. A refused or stale restore
leaves the draft, current Undo and performance intent intact.

## Browser surface and implementation stages

Add a compact **Saved setups...** entry beside the existing authoring tools,
using a native dialog rather than another persistent composer row. It lists
the selected checkpoint's setups first and can reveal the others. Provide
Save current, Rename/Update, Delete and Restore with an inspectable summary.
Keep unsaved names/edits on errors; stale updates/deletes return 409 and offer
a reread before discarding the editor. Held actions use `aria-disabled`, their
reason as title and an in-dialog response on a press.

1. **Cold storage and contract.** Implement strict schema, sidecar CRUD,
   request/profile validation and focused unit/API checks. No generation path
   or automatic restore is added.
2. **Explicit composer integration.** Add declarations in a focused classic
   script, wire only from `app.js`, and keep the stylesheet/script maps current.
   Close the modal before undo-aware prompt writes. Restore both boxes through
   `rewriteBox`/`writeBox`, references through `setReferences`, and run existing
   capability, tags, analysis, size, preview and persistence synchronization.
   Snapshot the whole previous draft and ownership flags for one whole-setup
   Undo, using the existing Reuse/Reset retirement rules; include conditioning,
   grid and Then refine in that snapshot. The first later user edit retires
   this Undo. Native Ctrl+Z still restores prompt text. Restore/Undo do not
   scroll the page or move controls under the pointer.
3. **Browser acceptance and documentation.** Add a scratch fake-server smoke
   and its AGENTS table row. Update usage, module maps and design notes with
   the final behavior; remove the TODO row only when these checks pass.

### Stage 1 implementation, 2026-10-28

`schemas_presets.py` separates the strict raw draft from capture transport.
`presets.py` stores the version-1 collection at `auradream.presets.json` beside
the database, with a separate process reservation, atomic replace and optimistic
item revisions. First reads create nothing. All items are type-checked on read;
unknown versions/fields, duplicate JSON keys/ids, malformed Unicode and oversized
files refuse without replacing the file. This new feature has no legacy filename.

The cold routes are `GET/POST /api/presets`, `GET/PUT/DELETE
/api/presets/{id}` and `POST /api/presets/{id}/validate`. Create/update captures
require `model_id` and `model_revision`; validation requires the saved item
revision and model revision. The existing model-request middleware holds the
selection through validation and saving. Reads/deletes remain available across
a checkpoint switch; all these routes bypass gallery bundle admission and the
generation gate. Updating a setup keeps its original checkpoint.

`preset_validation.py` uses the existing Generate/Grid request types, grid
helpers and `applied()` for an explanation separate from the stored draft.
The API supplies the backend's pure `applied` reader for live capability answers,
without calling `describe()` or CUDA. Null overrides follow the exact checkpoint's
profile; zero remains explicit. The nullable draft seed stays null. Random-grid
examples use a disposable seed of zero without drawing or returning a new base
seed. Grid Repeat becomes 1 only in the effective request, and Resume is false.
Validation checks the item revision again after calculation. It never submits
work or writes the draft back.

The answer keeps `saved` beside an `effective` base Generate request. `values`
are the ordinary requested grid axes; `effective_values` separately shows
normalized negative/Mu/True CFG values and `inactive` names inputs unused by
at least one cell. Grid definitions remain under `saved.grid`, rather than
becoming an expanded or resumable run.

The selected loader must be valid Eager/native and submitted Step cache intent
must be Off for capture/validation. Nonempty, invalid or unknown performance
intent refuses before a write; saved payloads cannot carry execution fields or
conditioning. Stage 1 added no browser entry or automatic restoration; stage 2
below owns explicit composer capture/Restore and whole-draft Undo.

Stage 1 validation passed **410 focused/regression tests** across the three
new files, existing grid/model-switch/snippet routes, execution admission and
resolution, and the classic-script guard. Repository Ruff and the pinned
Pyright gate passed (138 files, zero diagnostics). The installed environment's
Python ran these commands directly because `uv run` stopped before execution
with `uv trampoline failed to canonicalize script path`; dependencies and the
lockfile were unchanged. A fresh-process validation check imports no Torch,
Diffusers, Transformers or Hub client. No browser restoration, model load or
card trial was performed.

### Composer integration, 2026-10-28

`web/generation-presets.js` declares the setup dialog, with `initSetups()`
called only by `app.js`; `styles/setups.css` follows the snippet sheet.
**Examples > Saved setups…** opens it before Personal snippets, without a new
persistent caption control or startup read. The list shows this checkpoint
first, with **Show other checkpoints** to reveal the rest. **New** starts a
name, **Save current** creates an item, **Update from current** replaces its
payload, **Rename** keeps the saved payload and **Delete** removes only that
item. Failed/stale writes preserve the unsaved name. **Reload list** rereads
without replacing the editor; explicitly selecting again adopts the newest
item/revision. The [dated setup recovery review](review-2026-10-28-setup-list-recovery.md)
records the native read/selection and sole-item reselection fixes, with their
separate reproduction and qualification evidence.

Capture reads exact text and raw overrides directly rather than through
`collect()` or `gridPayload()`, which trim and suppress inactive inputs.
Unowned/blank sampling remains null, explicit zero remains zero, and the
nullable seed stays null. Save names active LoRA, references, LoRA weight axis
and Then refine as omitted-input holds. Current and unsaved picker performance
intent must both be valid Eager/native with Step cache Off.

Selecting checks saved/effective values for the summary. **Restore** performs
a fresh cold validation and binds the reply to the item/revision, model/revision,
server, complete composer/performance fingerprint and edit epoch. A change
while awaiting it asks for another deliberate Restore, leaving newer state
intact. Foreign checkpoints hold Restore and offer **Open Models**, which
opens the existing picker without switching or loading.

Restore closes the modal before `rewriteBox` writes Negative then Prompt,
restores ownership and raw inactive inputs, clears conditioning through
`setReferences` and the adapter controls, turns Then refine off and leaves
Resume off. The summary names these changes first. Existing size, capability,
tags/analysis, preview, submission and persistence synchronization then runs.
The restored grid remains unexpanded; ordinary preview and Run grid own its
effective axes, seed draws and request Repeat=1.

One Undo keeps the whole previous in-memory draft, raw sampling and ownership,
size selection, conditioning, grid, translation and Then refine controls/scale.
It uses the stable prompt-note overlay and existing Reset/Reuse retirement
rules. Panel, Then refine and performance edits retire it; deleted/binned
references leave its snapshot too. Both boxes retain native undo-aware writes.
Restore and Undo avoid Reuse's reveal/scroll path and queue no work.

Focused validation passed **243 tests** across the preset storage/validation/API,
grid API and classic-script guard. Repository Ruff and pinned Pyright passed
(138 files, zero diagnostics). The environment's Python again ran directly
because `uv run` failed before execution with the trampoline error recorded
above; no dependency or lockfile change was made.

### Browser acceptance, 2026-10-28

The new **172-step** `smoke-generation-presets.json` passed on Windows against
its fresh scratch FLUX.1 Krea [dev] fake. It checked two saved Generate/Grid
setups through reload, exact text and null/zero ownership, explicit Restore,
whole-draft Undo and native Ctrl+Z, conditioning/Then refine clear and Undo,
Resume off, held/invalid performance, stale item revisions and late draft/model
validation replies. A real switch of the fake checkpoint while the dialog was
open exercised the foreign-checkpoint hold and Models opener. Ordinary grid
preview/submission, modal keys/focus, page errors, and 1400x900/400x650 geometry
and scroll checks passed.

Five existing browser regressions also passed: `smoke-snippets`, `smoke-grid`,
`smoke-png-settings`, `smoke-execution-reuse` and `smoke-compact-setup`.
The six documentation-map tests passed with the updated script/style and smoke
maps. This completes the planned local storage/browser acceptance. No real
model load, card run, fit, quality or speed measurement was performed; these
checks establish interaction behavior against the fake.

## Completion evidence

CPU/API checks cover CRUD, Unicode/exact text, bounds and unknown/newer schema
refusal, malformed-file preservation, atomic failure, concurrent processes and
stale revisions. Check null/zero/profile sampling, seed limits and null seeds,
Batch/Repeat versus grid seeds, cross/random variables, invalid axes and
ordinary `applied()` parity. Prove no load, network, CUDA, job admission or
adoption of preview seeds occurs during CRUD/Restore validation. Include
checkpoint-switch races and held/invalid execution cases that leave the
previous draft intact.

The real-browser smoke covers saving two setups, reload, explicit restore,
whole-setup Undo and native Ctrl+Z, a different-checkpoint hold, stale item and
draft/model changes during validation, grid preview and a fake submission.
It also checks cleared conditioning/Then refine and Undo, Resume off, modal
key ownership and focus return, useful reasons, no page errors, and stable
geometry at 1400x900 and a narrow viewport. Run focused pytest, Ruff, the
pinned type gate, classic-script guard and relevant existing grid/Reuse smokes.
These establish storage and interaction behavior, not card fit, image quality,
performance or deterministic pixels across different runtime versions.
