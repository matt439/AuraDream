# Plan: export and merge personal snippets

**Status, 2026-10-28: stage 1 built and CPU-checked; stages 2-3 remain.** `UI#20` came
from the [TODO audit](review-2026-10-28-todo.md). This extends the completed
[personal snippet feature](plan-prompt-snippets.md); its implementation and
acceptance are CPU/browser work, without a model or GPU run.

## Purpose and current boundary

Provide an explicit way to keep a portable copy of personal snippets and
merge one into another AuraDream installation without editing sidecars.
`backend/snippets.py` currently stores exact text beside the database, with
an independent cross-process lock and per-item revisions. The CRUD API in
`api_snippets.py` has no export/import operation. Gallery bundles and
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
