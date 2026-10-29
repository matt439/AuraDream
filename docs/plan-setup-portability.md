# Portable generation and grid setups

**Status:** `UI#25` selected for independent source implementation on
2026-10-28. Stage 1 (cold codec and explicit merge planner) and stage 2 (the
atomic cold export/preview/import API, 2026-10-29) are implemented; stage 3,
the browser merge dialog, remains. No user-visible action is active yet.
`GAL#12` retains its own incomplete qualification boundary.

## Ranked selection

1. Portable named Generate/Grid drafts: the existing sidecar is outside
   gallery bundles and image reuse cannot transfer unexpanded named grids.
   This fills a concrete gap with a small native format and explicit merge.
2. Fake-only credential-provider refusal for reused enabled settings: useful
   isolation, while current genuinely new disabled scratch galleries are cold.
3. Guard manual fake-server `--fresh` target ownership: useful hygiene; the
   ordinary runner already creates a new temporary directory and omits that
   flag. No destructive reproduction is needed.

Choose the first for direct user value and bounded reversible implementation.
The native format is `auradream.generation-presets`, integer version 1, with
download name `auradream-generation-setups.json`. ComfyUI's official workflow
JSON documentation supplies a portable-recipe precedent, not format
compatibility or evidence of measured user demand (source linked below).
The user's autonomous new-work instruction permits this selection without a
routine product-choice question; explicit existing deferrals remain intact.

## Contract

Use a separate native UTF-8 JSON document with exactly `format`, integer
`version: 1` and ordered `items`. Require the chosen exact format literal;
reject booleans/floats as versions, unknown versions/fields, duplicate JSON
keys, nonfinite numbers, invalid Unicode and excessive nesting. Portable items
contain **only `PresetFields`: `name`, exact `model_id`, typed raw `payload`**.
Emit the complete payload, including nulls/default fields, using the current
schema's semantics. Never serialize `Preset` or capture-transport models.

Retain the existing 100-item, 80-code-point name and 8-MiB portable/final-store
bounds. Preserve exact Prompt/Negative and axis/variable text, array/item order,
cross/random mode, null sampling ownership, explicit zero, nullable seed,
Batch/Repeat and budget. Existing name trimming/type/Unicode validation stays
authoritative. Exclude local ids/revisions, `model_revision`, `step_cache`,
conditioning, image/job/run identity, Resume, loader and execution fields at
every nested level; reject rather than strip them.

Import stores raw drafts only. It neither certifies restorability nor calls
selection/backend appliers, expands grids, draws seeds, contacts the Hub,
switches/loads models or queues work. A foreign or currently unavailable exact
checkpoint id may remain in the library. Existing explicit Restore must still
perform selected-checkpoint/revision/capability/profile and execution checks.
Do not use `preset_validation.validate_payload`'s generic unknown-model fallback
as proof a foreign setup can run (`api_presets.py:66–86`, `preset_validation.py`).

## Identity and merge behavior

Preserve existing legal duplicate names. Use `(NFC/casefold(name), EXACT
model_id)` only to identify possible conflicts; the model id is case-sensitive
and is never normalized. Generate/Grid is payload, not a separate namespace.
The same name on another checkpoint is a different key.

- Compare exact matches by the full validated name/model/payload, including
  raw strings and null ownership. An exact export/reimport defaults to Skip.
  A differing local conflict also defaults to Skip; no local candidate defaults
  to Add. Show every matching local id/revision, without first-match selection.
- Add is allowed only without a local conflict. Add as copy is explicit and
  may retain a legal duplicate name or edit the name within existing bounds.
  Only Copy may edit a name. Replace requires one explicit matching local
  id/revision on the **same checkpoint**; preserve that id, checkpoint and row
  position. An explicit exact Replace remains a revisioned edit, matching the
  existing snippet contract. Skip changes nothing.
- Keep each incoming file index distinct. Flag incoming collisions after any
  Copy-name edits; never collapse/deduplicate them. As with snippets, separate
  nonconflicting-to-local rows may each Add even when their incoming keys match;
  show that group and all choices. Reject duplicate/out-of-range decision
  indices and two replacements of one local id. Append additions in file order.

Preview may fill missing choices with these defaults. Commit requires exactly
one complete decision for every file index. Empty/all-Skip plans cannot commit.
Check final count and exact sidecar serialization size, using 32-hex placeholder
addition ids until persistence assigns real UUIDs.

The library digest covers ordered complete local rows, including ids/revisions;
the file/payload digest covers canonical validated format/version/items and all
raw payload data. These are stale-content checks, not operation receipts or
idempotency keys; JSON whitespace can be equivalent. Keep original file text
and file/session identity in the UI. Under the existing **preset-path lock**,
reread the library, recheck both digests and replacement revisions, recompute
the plan and bounds, then use one existing atomic writer. No partial merge,
automatic rebase/retry, name-based overwrite, local deletion or library replace.
The existing `auradream.presets.json` path/reservation remains unchanged; do not
invent database-filename isolation or a sidecar migration in this feature.

## Three reviewable stages

1. **Cold codec/planner.** Add preset-specific portable/request/decision/preview
   models and pure export/parse/digest/merge functions. Revalidate constructed
   model inputs, avoid UUID allocation or mutation during planning, and match
   `presets._write` encoding exactly. Meaningful tests cover Generate/Grid raw
   round trips, null versus zero, Unicode/axis order, forbidden fields at all
   levels, bounds, foreign models, cross-model name separation, legal duplicate
   candidates, incoming file collisions, exact/default/Copy/Replace/Skip,
   revision refusal, ordering, no-op and input immutability.

2. **Atomic cold API.** Add export, preview and commit routes analogous to the
   snippet transfer routes. Register static GET export **before** existing
   GET `/{item_id}`. Bound streamed request bytes before FastAPI JSON parsing,
   without trusting Content-Length; keep a separate payload bound and a small
   envelope allowance. Return fixed import validation/refusal text, attachment
   JSON and no-store export headers. Add only exact transfer shapes to
   `GalleryAdmission` exemptions; keep `depends_on_model` capture-only matching.
   Import must not call ordinary capture POST/PUT validation. Tests exercise
   actual scratch atomic writes, stale whole-list/target/file checks, complete
   choices, concurrent commits/CRUD, lock contention, malformed destination
   preservation, write/fsync/replace cleanup, first-read coldness, existing
   sidecar path sharing, and model/switch/gallery/vault sentinels. Export all
   saved checkpoints from one coherent snapshot, not the UI's filtered view.

3. **Saved-setups UI.** Add Export/Import and a separate setup-specific merge
   dialog/module; leave Snippet UI unchanged. Preserve the existing library
   layout and text-only rendering. Label each incoming checkpoint and raw
   payload; use row-local candidate numbers. Do not call `setupSummary` for
   import rows: it writes the parent editor and reads the composer. File/name/
   choice changes invalidate the preview; current file/session tickets reject
   late reads/previews/downloads. Export's dialog-session lifetime must be
   independent of Reload's list ticket, as the snippet precedent already is.

   Success updates the saved list only, keeping the old editor name/selected
   revision/dirty draft/composer/Undo. Hold actions using a replaced/removed
   selected revision until explicit reselect/New; never auto-adopt. Coordinate
   pending ordinary writes/imports and Close/Escape. A transport/body-read/5xx
   commit failure is uncertain: no resend; require fresh list + preview, retain
   stale replacement choices for explicit review, and clear/reconfirm each old
   Copy action before another commit. Also keep ordinary setup writes held
   behind fresh-read plus explicit selection/New acknowledgement. Reopening,
   Reload, preview or a successful unrelated import cannot silently acknowledge
   an earlier uncertain ordinary write. Known 422 stays correctable; 409 needs
   refreshed choices. New/selection before a held GET finishes cannot acknowledge
   recovery or let the late reply replace the draft.

   Qualify with owned scratch browser files/data: mixed checkpoints/modes,
   actual export bytes, duplicate choices, raw round trip, stale/uncertain
   commits, held-read/New/close races, unchanged composer and no model/queue
   calls, native focus/modal keys, stationary controls and wide/narrow/short
   geometry. Rerun affected setup/library recovery/Unicode checks plus classic
   scope/docs and relevant CPU/API checks. No card or real-gallery acceptance.

## Parsing and compatibility

Use a small bounded syntax reader for the new native format, with explicit
byte/depth limits and domain-specific errors. Keep the existing snippet codec
unchanged. A proposed consolidation was not applied: two repository Git
symbol searches encountered an ownership/trust denial, and an ensuing
filesystem search for the same symbols was a boundary-handling mistake. Those
queries are stopped, no trust setting changed, and no evidence from that
fallback is needed for the new preset-only path. Small syntax duplication is
preferable to tying this independent feature to a stopped operation.

Do **not** silently route `presets._read` through UTF-8-only portable parsing:
its existing `json.loads(bytes)` accepts UTF-16/32 and already rejects duplicate
fields. Preserve sidecar compatibility with explicit UTF-16/32/read-edit
regressions if shared parsing touches that path; preserve duplicate/deep-file
refusal and malformed-file retention for snippets. New portable files may be
strict UTF-8 without changing existing sidecar formats.

Official context: ComfyUI documents saving/sharing workflow JSON independently
of generated media: https://docs.comfy.org/basic-concepts/workflow . This is a
portable-recipe precedent, not format compatibility or measured AuraDream demand.

UI#20's older denied screenshot reinspection remains OPEN. This feature must
not reread that target, regenerate its screenshots to bypass the denial, or
claim its missing visual acceptance. New setup-feature evidence has its own
owned artifacts and scope. All explicit deferrals and credential/card gates
remain unchanged.

## Stage 1 local qualification

The five-new-file preset-only codec/schema/reader/test slice is implemented
and independently reviewed. The proposed old snippet extraction remains
unapplied. All existing preset and snippet readers are unchanged.

On 2026-10-28, serial targeted codec/reader and existing saved-setup storage,
validation and API checks passed **293 tests in 4.99 seconds**. The fresh
isolated `-I -B` process exercised export/parse/digests/planning/preview while
denying application, model/grid, store, vault and heavy dependency imports;
there were no attempted forbidden imports. Independent store/UUID sentinels
and input-immutability checks passed. Scratch UTF-16/32 sidecars retained
read/edit behavior and malformed duplicate-key files were preserved.

No-cache Ruff passed. The installed pinned Pyright 1.1.407 gate analyzed
146 backend/tools files with zero errors or warnings. A pre-existing FastAPI
TestClient/httpx deprecation warning remains separate. No transfer route/UI,
model/backend loading, credential access, real gallery or card qualification
is established by these checks. Stage 2 adds atomic persistence/HTTP; stage 3
adds the explicit merge UI and its own browser acceptance.

## Stage 2 local qualification

Routes in `backend/api_presets.py`, mirroring the snippet transfer routes:

- `GET /api/presets/export`, registered before `GET /api/presets/{item_id}`:
  one `presets.read` snapshot of every checkpoint's setups through
  `preset_portability.export_portable`, as `application/json` with
  `Content-Disposition: attachment; filename="auradream-generation-setups.json"`
  and `Cache-Control: no-store`. It never writes or creates the sidecar.
- `POST /api/presets/import/preview` (`PresetImportRequest` to
  `PresetImportPreview`): read-only plan with safe defaults.
- `POST /api/presets/import` (`PresetImportCommit` to `PresetCollection`):
  `presets.import_collection` holds the existing preset-path reservation,
  rereads the sidecar, rechecks the library digest (409), file digest (422),
  complete distinct choices (422) and, through
  `plan_merge(require_complete=True)`, replacement revisions (409), final count
  and exact encoded size, then assigns UUIDs and calls the existing `_write`
  once. No partial merge, retry or capture validation.

Both import routes stream at most `MAX_PORTABLE_BYTES + 64 KiB` raw bytes before
the strict UTF-8 reader, ignoring Content-Length (413 "The setup import request
is too large."); the portable payload keeps its own 8-MiB bound. Parser and
schema refusals are fixed texts ("The setup import fields are invalid.", the
codec's messages) and never echo the file. `GalleryAdmission` exempts exactly
`/api/presets/export` and `/api/presets/import(/preview)`; `depends_on_model`
stays capture/validation-only, so a switch or maintenance never holds transfer.

`tests/test_api_preset_portability.py` (72 tests) covers export headers,
contents and coherent snapshot, route order, cold missing-file preview,
candidates/exact ids/file-index collisions, explicit Replace/Copy/Add order,
one writer call without capture validation, stale whole-list/target/file
digests, replay, incomplete/duplicate/extra/all-Skip choices, cross-checkpoint
and double replacement, no-op holds, raw parser refusals, stream and payload
byte bounds, malformed destinations (including duplicate keys) preserved,
shared sidecar path, scratch round trip, concurrent imports and import-vs-CRUD,
write/fsync/replace cleanup, lock contention, count/byte rechecks under the
lock, foreign checkpoints stored raw, origin refusal and
model/switch/Hub/queue/selection/gallery-maintenance sentinels.

On 2026-10-29 on Windows: the new file plus `test_preset_portability.py`,
`test_presets.py`, `test_api_presets.py`, `test_preset_validation.py`,
`test_api_snippet_portability.py` and `test_strict_json.py` passed
**421 tests in 24.67 seconds**; the full suite (`-n auto`, excluding the
environment-only `test_browser_startup_wait.py`) passed **7667, 8 skipped**.
No-cache Ruff passed; the pinned Pyright gate analyzed 146 files with zero
errors or warnings. No browser UI, model loading, card or real-gallery
qualification is established by these checks; stage 3 owns the UI.
