# Plan: export and merge personal snippets

**Status, 2026-10-29: stages 1-3 built, CPU/browser-checked and visually inspected; complete.** `UI#20` came
from the [TODO audit](review-2026-10-28-todo.md). This extends the completed
[personal snippet feature](plan-prompt-snippets.md); its implementation and
acceptance are CPU/browser work, without a model or GPU run.

## Purpose and current boundary

Provide an explicit way to keep a portable copy of personal snippets and
merge one into another AuraDream installation without editing sidecars.
`backend/snippets.py` currently stores exact text beside the database, with
an independent cross-process lock and per-item revisions. The CRUD API in
`api_snippets.py` now also provides snapshot export, stateless merge preview
and explicit atomic import; the browser integration remains stage 3. Gallery bundles and
automatic gallery backups do not include this sidecar, as recorded in
[the design notes](design-notes.md#personal-prompt-snippets).

This is snippet portability, not a change to gallery backup scope. Imported
text is data and is inserted only through the existing explicit Append or
Prepend action. It is never a prompt instruction for the app or the agent.

## Portable format and bounds

Use a UTF-8 JSON object with exact fields `format`, `version` and `items`.
Version 1 has `format: "auradream.prompt-snippets"`; every item has only
`name`, `text` and `target` (`prompt` or `negative`). Export saved items in
their current order, preserving their exact validated text. Omit local ids,
revisions, paths, current drafts, gallery records and settings. Imported
items receive local identity only when committed.

Reuse `SnippetFields` validation and the existing limits: at most 100
items, 80 characters per name, 16,000 per text and an 8 MiB encoded file.
Check the incoming byte limit before parsing; reject unknown/newer versions,
unknown fields, duplicate JSON object keys, malformed UTF-8/JSON, invalid
targets, invalid Unicode and nesting deeper than 16 levels through fixed
errors. Do not
interpret an old sidecar as the portable format or silently strip its fields.
An empty export is valid; importing it is a no-op, never a request to clear
the destination. Check the final merged collection's count and encoded
sidecar size before committing, not just the incoming file's bounds.

The export is a plain-text download named `auradream-snippets.json`. Its UI
states that the file contains personal prompt text. No automatic upload,
sharing, encryption, import-at-startup or folder watching is added.

## Preflight and explicit merge

Import first previews a merge against a fresh saved collection. The server
returns an opaque collection revision covering all local ids, item revisions,
ordering and content, plus a digest of the validated portable input. Keep
the existing storage version readable; this collection revision can be
computed from the validated snapshot rather than replacing item revisions.

Every incoming item is addressed by its file index, so files containing
duplicate names remain unambiguous and an ordinary export can round-trip.
Classify exact matches by the full validated `(name, text, target)` tuple.
Use an NFC/case-folded name plus target only to identify potential conflicts;
it does not authorize replacement or change the original text.

- An exact local match defaults to **Skip**. Show which local item it
  matches. A deliberate **Add as copy** remains available.
- A name/target with no local collision defaults to **Add**.
- A name/target collision with different text defaults to **Skip** and
  shows both texts. The user may choose **Add as copy**, optionally with an
  edited valid name, or **Replace** one explicitly selected local item.
  Existing duplicate names are legal, so the UI must not select the first
  matching row on the user's behalf.
- Show collisions within the incoming list too. Keep file-index decisions
  distinct, and show all copies that the final plan would add. Reject two
  replacement decisions targeting the same local id. No text concatenation,
  name-based overwrite, local deletion or collection replacement is implied.

The preview lists Add/Replace/Skip totals and each decision. **Import** is
held when the plan has an unresolved/invalid choice, exceeds final limits,
or changes nothing. Editing choices or names invalidates the shown preview;
recompute it before allowing Import. Canceling preview writes nothing.

At commit, supply the same portable payload, its digest, the collection
revision and the chosen file-index decisions, including target ids/item
revisions for replacements. Under the existing snippet lock, reread the
destination, validate both input and decisions, compare the collection
revision, and recompute the exact merge before one atomic write. A concurrent
CRUD/import change returns 409 with no mutation. Retain the file and choices
for a fresh preview, but never retry or rebase replacements automatically.
Adds use fresh server-owned ids at revision 1; replacements retain the local
id and increment its item revision. Skips preserve their original rows/order;
append additions in file order and retain replaced rows' positions.

This stateless preview/commit protocol needs no server upload directory or
unbounded preview cache. A lost commit response is uncertain to the browser:
reload and preview again, rather than blindly resending Add as copy. Exact
matches can then show what is already saved. Do not promise idempotency from
the file digest alone.

## API and dialog integration

Proposed model-free routes are `GET /api/snippets/export`,
`POST /api/snippets/import/preview` and `POST /api/snippets/import`.
Register static routes before `/{item_id}` and keep the current CRUD URLs
working. The preview/commit JSON envelope carries the portable object as
`payload`, not an escaped JSON string, and at most 100 bounded decisions.
Bound raw request bodies to 8 MiB plus 64 KiB before parsing; validate the
portable object's encoded size separately against 8 MiB. Reject oversized
requests without returning their text.
Export reads one validated snapshot and returns only saved items. Failure to
read a malformed/newer local sidecar refuses both export and import; neither
operation replaces it with an empty collection.

Put the format/parser/merge planner in `backend/snippet_portability.py`,
with strict request/response models beside existing snippet schemas. Keep
atomic persistence in `snippets.py`; do not implement another writer. The
preview and commit remain available during gallery maintenance and do not
acquire a gallery reservation, load lock or GPU gate.

Add **Export…** and **Import…** inside the existing Personal snippets dialog,
without another composer caption button. Export downloads the current saved
list, not an unsaved editor. Import opens a file chooser and an explicit
merge preview; showing or choosing a file does not save anything. Use text
rendering for file names, snippet names and text previews. Display errors in
the dialog, preserving the selected file, choices and unsaved snippet editor.
After a successful import, refresh the saved list without replacing that
editor draft; an editor whose selected item changed must re-read its revision
before it can save. Closing the import preview returns focus to Import;
closing the snippet dialog follows its existing focus/scroll contract.
Its modal keys must not generate underneath it. Pending writes and held
Import use the established reason/title behavior.

## Stages and completion criteria

1. **Portable codec and pure merge plan.** CPU tests prove exact Unicode and
   multiline round trips, both targets, empty collections, saved-order
   preservation, every boundary, duplicate-key/nesting/version refusal,
   multiple existing/incoming names, exact-match skips and each conflict
   decision. Verify that file ids/revisions/paths cannot enter local identity
   and that malformed inputs produce no sidecar or prompt mutation.
2. **Snapshot export and atomic commit.** Add the bounded routes and whole-
   collection revision. API/storage tests cover independent databases,
   export during concurrent CRUD, unrelated concurrent edits invalidating
   a preview, stale targeted revisions, conflicting decisions, final count/
   byte overflow, lock contention, write/fsync/replace failure and unreadable
   local-file preservation. Round-trip one scratch collection into another;
   repeat it to show default exact skips. Assert one atomic commit and no
   partial additions on any refusal. Block model/gallery/vault dependencies.
3. **Browser merge workflow.** Extend the scratch snippet smoke or add its
   own authoritative smoke row. Download and inspect the actual exported
   bytes, choose a valid file, inspect Add/Replace/Skip and commit the chosen
   plan. Cover conflicting/duplicate names, unchanged/empty import, malformed
   and oversized files, stale preview, uncertain response, cancellation and
   dirty-editor retention. Prove import leaves Prompt/Negative unchanged;
   explicit insertion afterward still supports native Ctrl+Z and holds an
   unsupported negative. Check keyboard ownership, held reasons, focus return,
   stationary controls and containment at 1400x900, 400x650 and a short
   desktop window. Re-run existing snippet/author-example compatibility,
   classic-scope/docs-map checks, Ruff and affected-source pinned types.

Record actual results here when implemented. A synthetic round trip proves
portability and merge behavior, not live user data recovery or usability.

## Stage 1 implementation, 2026-10-28

The cold codec and strict portable/decision models are implemented in
`snippet_portability.py` and `schemas_snippet_portability.py`. UTF-8,
duplicate-key and depth checks precede parsing; input and saved-collection
digests cover validated content and ordered local identity. The pure plan
defaults collisions to Skip, identifies incoming duplicates by file index,
and requires a specific local id/revision for Replace. It retains positions
and checks final sidecar count and encoded size using same-width placeholder
ids; new identities remain the future atomic writer's responsibility.

CPU tests cover the stage-1 acceptance cases, including exact text/order,
both targets, empty imports, malformed encodings and fields, normalized
duplicate names, stale/ambiguous replacements and combined byte overflow.
Ruff and pinned Pyright passed. Routes, atomic import and the browser merge
workflow are still stages 2-3; no real user sidecar or prompt was changed.

## Stage 2 implementation, 2026-10-28

The three model-free routes are implemented. Export reads one validated
snapshot and downloads only saved portable fields with no-store caching.
Import requests are byte-bounded while streaming and parsed for UTF-8,
duplicate keys and depth before FastAPI validation. Local sidecars now
use that strict parser too; malformed/newer files remain untouched.

Commit requires one explicit decision per file index, the input digest
and the whole-collection revision. The existing snippet reservation covers
reread, revalidation, recomputation and one existing atomic writer call.
Fresh ids are assigned only to committed additions; replacements preserve
position/id and increment revisions. Empty/unchanged plans refuse writes.

Scratch tests cover snapshot export during CRUD, unrelated stale edits,
competing imports, repeated commits, stale/conflicting replacement targets,
complete choices, corrupt/oversized inputs, raw streaming bounds and
write/fsync/replace failures. A scratch-to-scratch round trip retains text
and order, assigns fresh ids and defaults repeat imports to exact skips.
Model, queue and credential access stay blocked in tests; the routes bypass
gallery maintenance and switching. Browser integration/acceptance remains
stage 3, and no real gallery, snippet file or credential was used.

## Stage 3 implementation and browser evidence, 2026-10-28

**Export.** and **Import.** now live inside Personal snippets. The sibling
native merge dialog renders incoming/local text safely, identifies every
file item separately and requires an explicit local candidate for Replace.
Choice edits retire the old preview; no operation touches Prompt/Negative.
The raw selected JSON is retained until the server parses it, preserving
duplicate-key refusal. Copy names use the backend's Unicode code-point bound.

Import refreshes the saved list while keeping the editor text and old
selected revision. Save/Delete hold changed or removed selections; explicit
reselection reads the new revision. Stale/uncertain responses retain the file
and choices but never resend. Recovery reloads candidates without rebasing
replacement revisions. A Copy whose first commit may have succeeded needs a
new explicit action selection, so Preview alone cannot authorize duplicates.

The final **183-step `smoke-snippet-portability` passed on Windows** against
a fresh FLUX.1 Krea [dev] fake on 7934 (offset by 100 for the run). It checks
actual export-response Blob bytes/order/identity omission, both targets,
raw duplicate-key/UTF-8/version/size errors, exact skips, duplicate incoming
and existing names, explicit replacement/copy choices, final identities,
stale collection recovery, successful commits with lost Add/Copy responses,
80/81-code-point copy names, pending-close and late-preview ownership,
dirty-editor retention, focus, modal keys, unchanged prompt boxes, held
negative insertion and native Ctrl+Z. The existing `smoke-snippets` regression
also passed, including author-example/menu compatibility and caption geometry.
Both ran sequentially with one worker and independent scratch galleries.

Containment and fixed-action checks pass at **1400x900, 400x650 and
1400x600**, including field target heights. Three screenshots from the first
182-step passing run were inspected. They exposed short native selectors;
the selectors now use the app's field size/color tokens, and the final run
passes a new target-height/alignment assertion. **Final screenshot
reinspection is incomplete:** `Get-ChildItem` of the final run's task-local
`auradream-smoke-artifacts-4vl6i3n2` directory was denied. That read was not
retried through a different route. Browser interaction acceptance passed;
the remaining visual reinspection keeps `UI#20` open rather than claiming a
fully inspected final set of images.

The old Windows `IAppCapability 0x80070005` denial was reconciled before any
launch: [tooling](plan-tooling.md#browser-smokes-in-ci) records ordinary Edge
access restored on 2026-10-27 without flag/security changes, and the current
[saved-search acceptance](plan-saved-searches.md#browser-acceptance) records
later successful qualification on this branch. The approved runs used that
same ordinary harness, with no alternative browser route or WSL operation.

The combined snippet/API/static/docs group passed **197 checks**, including
the syntax/shared-scope gate over all **57 classic scripts**. Ruff and pinned
Pyright pass (143 Python sources, zero diagnostics; Python sources unchanged
since the API stage). Independent source review is clear. Browser fixtures
use synthetic File/DataTransfer selection and intercept the real export Blob
before an OS Downloads write; native OS chooser/download behavior and real
user-data recovery were not tested. No real gallery, snippet sidecar,
credential, model, GPU operation or release admission was involved.

## Final screenshot inspection, 2026-10-29

A fresh run of `smoke-snippet-portability` passes (12 s), and its three new
screenshots were inspected:

- `snippet-portability-wide.png` (1400x900): the hostile
  `<img src=x onerror=…>` file name and item name render as text. The row
  list scrolls inside the dialog with its own scrollbar. Action and
  Copy-name/Replace selectors use the field tokens, and the Preview/Import
  footer stays in view.
- `snippet-portability-narrow.png` (400x650): fields stack one per line, and
  the footer and its note stay visible without page scroll.
- `snippet-portability-short.png` (1400x600): the list shrinks to keep the
  footer in view.

No clipping or overlap is observed. Each item's preserved leading space
(" exact") is the snippet's own text, shown verbatim. `UI#20` is closed.
