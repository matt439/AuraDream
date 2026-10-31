# Plan: named gallery searches

**Status: done 2026-10-28** (`GAL#7`).

**Status, 2026-10-28: built; local CPU/browser acceptance recorded below.**
`GAL#7` came from the [TODO audit](review-2026-10-28-todo.md). The implementation
uses scratch galleries for acceptance; no model, GPU run or live gallery is needed.

## Purpose and current behavior

Save a useful gallery filter under a name, then apply it in another session
without rebuilding its search terms and ticks. Matt reports frequent gallery
browsing, Starred, All models, prompt searches and searches from image facts
([preferences, E1-E9](design/preferences.md#e---finding-and-organising-images)).
This is a convenience hypothesis, not an observed time saving.

Today `web/gallery.js:loadGallery` reads the search, Starred and All models
controls directly. `backend/search.py` owns their query grammar;
`AppServices.model_scope` supplies the running checkpoint's exact scope.
`web/gallery-search.js` adds facts to the existing search, and `runs.js`
opens recent runs. None stores named filters. The rules and history are in
[gallery scope](design-notes.md#the-gallery-is-the-running-checkpoints) and
[search grammar](design-notes.md#the-search-grammar).

## Saved definition and limits

Each entry stores only a name, query text, `favorites` and `all_models`,
plus its server-owned id and optimistic item revision. It does not capture
matching image ids, picks, counts, a page cursor, a job, sheet axes or the
selected model. Query terms such as `run:`, `id:` and `model:=` retain their
ordinary grammar; a saved query is not an image collection.

- A name is 1-80 characters after trimming, with no control characters.
  Names are unique after Unicode NFC normalization and case folding;
  saving a duplicate offers an explicit update of that named entry.
- Query text is at most 4,000 characters and valid UTF-8. Preserve its text
  rather than rewriting or broadening it. Empty text is allowed: Starred or
  the checkpoint scope alone can be useful.
- Store at most 50 entries, in a strict version-1 envelope capped at 1 MiB.
  Reject unknown fields/versions, duplicate ids or names and invalid types.
  Run `search.split_search` for its existing filter validation, without
  executing SQL or introducing a second query language. Free text remains
  a browse query; saving it does not make it a valid Delete unstarred scope.
- `all_models: false` means the checkpoint selected when the entry is
  applied, not the checkpoint selected when it was saved. Show that wording
  in the dialog. `date:today` and `date:yesterday` also retain their normal
  apply-time local-day meaning.

Use a new database-scoped sidecar,
`auradream.saved-searches-<database.name>.json`, beside the database. The
database filename keeps two databases in one directory independent. This is
a new AuraDream feature with no legacy file alias. Add the sidecar and its
temporary-file pattern to Git ignores. No bookmark stores image pixels or
generation settings beyond what the user typed in the query.

## Persistence and API

Follow the bounded atomic storage pattern in `backend/snippets.py`, with
an independent cross-process lock for this sidecar. A first read creates
nothing. Under the lock, read and validate the existing file before any
mutation; unreadable/newer files stay untouched. Flush/fsync a same-directory
temporary file, atomically replace, and clean only the owned temporary.

Proposed routes are `GET /api/gallery/saved-searches`, `POST` on that
collection, and `PUT`/`DELETE /api/gallery/saved-searches/{id}`. Updates and
deletes require the last read item revision. A stale write is 409; it keeps
the newer entry and the browser's unsaved editor. The response supplies the
saved list and limits. Reads and writes are model-free and do not take the
gallery maintenance reservation or the GPU gate; they remain available
during bundle work. Use fixed validation messages without echoing arbitrary
query text into logs or error responses.

Keep the storage/validated models/routes separate from the gallery's SQL
readers. `saved_searches.py`, `schemas_saved_searches.py` and
`api_saved_searches.py` are the intended owners. Register their router
through the existing service wiring; update the module map when built.

## Interaction and scope

Add **Saved searches…** to the gallery footer, opening a native dialog with
the saved list, Name/Query/Starred/All models editor, **Use current filter**,
New/Save/Delete and **Apply to gallery**. Keep the search header and composer
height intact; measure the footer with its count, Picked and Select controls
before accepting the placement. Read the list only when opened. Render names
and query previews as text. Selecting an entry only fills the dialog.

**Use current filter** copies the current live-gallery fields into the
dialog without saving. Hold it in the recycle bin with a reason: bin scope
is not part of this version. The dialog can still manage and apply saved
entries there. Save/Delete affect definitions only; deleting an entry does
not change the current gallery filter or any image.

**Apply to gallery** closes the dialog and performs one shared filter
transition: leave the bin and current sheet first, write all three controls,
then reset/reload the gallery and Runs once. Cancel a pending search debounce
and invalidate old gallery pages through the existing epoch. A sheet's
return filter must not overwrite the applied entry afterward. Reuse the
existing query and model-scope paths; opening a bookmark never switches or
loads a checkpoint. Keep an explicit `model:=...` query as written even if
the current-checkpoint scope makes its result empty; the existing other-model
count explains that result instead of silently changing All models.

Existing picks continue to follow the selection contract: they survive a
filter change, remain visibly counted, and actions still prefer those picks.
They are never saved/restored by a bookmark. When picks exist, the apply
status also says that they remain selected. Applying starts no generation,
selection, star/bin operation, export or deletion. Those actions retain
their own previews and confirmations over current server state.

An edited filter is an ordinary filter; it does not silently update the
saved definition. Do not auto-apply the last bookmark at startup or persist
the current ticks as a side effect of this feature. Pending writes and held
actions use the established reason/title behavior. Escape/Close restore the
connected opener without scrolling; the modal owns Ctrl/Cmd+Enter.

## Stages and completion criteria

1. **Definition store and routes.** Implement strict models, bounded reads,
   atomic CRUD, independent locking and optimistic revisions. CPU/API tests
   prove two-database isolation, cold/no-file reads, Unicode/name/query limits,
   grammar refusal, duplicate names, malformed/newer-file preservation,
   concurrent create/update/delete, stale revisions and write/replace failures.
   Confirm no route loads a model or waits on gallery maintenance.
2. **Dialog and single filter transition.** Implement declaration-only
   `web/saved-searches.js`, initialized by `app.js`, and the native dialog.
   Reuse existing tokens and CSS cascade. Add a scratch fake-server smoke to
   the authoritative AGENTS table and runner. Cover save/edit/delete/reload,
   Use current filter, Apply/Cancel, held reasons, dirty-editor/error retention,
   modal keys, and focus return. Verify late reads after Close/reopen do not
   overwrite newer edits.
3. **Scope and layout acceptance.** With two synthetic checkpoints, prove
   current-model versus All models behavior, Starred and free text, empty
   matches, a model switch, relative dates and a new result joining a saved
   search. Apply over an in-flight page/debounce, a sheet and the bin; assert
   old results/return filters cannot win. Preserve existing picks and their
   outside-view count; assert Apply makes no mutating action request. Check
   1400x900, 600x900, a short window and Gallery beside at the wide tier, with
   no new header row, clipped controls, automatic scroll or controls moving
   under a resting pointer. Run focused gallery/search/selection/sheet/bin
   compatibility smokes, docs-map/classic-scope checks, Ruff and the pinned
   type gate for the affected sources.

Record actual implementation and checks here before removing the TODO row.
Do not report usability improvement or real-gallery acceptance from fake data.

## Implementation, 2026-10-28

All three stages are implemented. `saved_searches.py` owns the strict version-1
sidecar, bounded UTF-8 reads, unique NFC/casefold names, independent process
reservation and atomic flush/fsync/replace. It rejects duplicate JSON keys,
unknown versions/fields and invalid queries through `search.split_search`.
The API returns fixed validation/refusal text and a duplicate's opaque id so
the browser can offer an explicit update using the current item revision.
Reads create no file, and all CRUD bypasses model and gallery admission.

`saved-searches.js` declares the native dialog; only `app.js` wires it. Its
editor retains unsaved text and old revisions on failures and Reload list.
Reload deselects the list, allowing a clean single-item editor to explicitly
adopt a refreshed revision by selecting that item again. Duplicate replacement
has its own explicit button. Read tickets and edit epochs prevent late reads
from replacing a newer editor or a reopened dialog. Writes hold their controls
with reasons until the request ends.

Apply uses one `applyGalleryFilter` transition. It leaves the bin, closes the
sheet without an intermediate reload (`setSheet(..., {reload: false})`), retires
the sheet return filter and search debounce, then writes all three fields and
resets the gallery epoch and Runs once. Existing picks and their hidden count
remain. Apply may use the editor without saving; its title explicitly names
that scope. Nothing is applied at startup or by selecting a definition.

The footer uses the full **Saved searches…** label where space permits, and a
named search glyph below an 800px footer width. Narrow and wide Gallery beside
spacing retains the existing action targets and selection count. The modal
keeps a fixed feedback slot so conflicts and status changes do not move its
controls. Its body scrolls in a short window while Close and Apply stay visible.

### CPU and API acceptance

The **61** focused store/API tests pass, including two-database isolation,
strict fields/Unicode/query grammar, duplicate names at capacity, corrupt/newer
file preservation, owned-temp cleanup on fsync/replace failures, stale writes,
concurrent threads and native cross-process CRUD locking. Route tests prove
CRUD remains available during gallery maintenance and a model switch without
loading a model or entering the GPU gate.

The existing gallery, snippets, setup API, bundle API and gallery-lock checks
also pass (**150 tests**). New tests plus docs-map, classic-script and smoke-runner
checks pass (**114 tests**). Repository Ruff and pinned Pyright pass (141 Python
sources, zero diagnostics). The environment's pytest executable trampoline
fails before test startup; `uv run --group dev python -m pytest` runs the same
installed suite successfully, with no dependency or lockfile changes.

### Browser acceptance

The **209-step** `smoke-saved-searches.json` passes on Windows against a fresh
scratch FLUX.1 Krea [dev] fake on port 7933. It covers CRUD and persistence on
a new page, safe text rendering, explicit duplicate replacement, failed/stale
writes with the editor retained, clean single-item Reload/reselection and a
late list reply after Close/reopen. Native Ctrl/Cmd+Enter, Escape, held reasons,
opener focus and page-scroll checks pass.

Two synthetic checkpoints prove current-model versus All models scope, Starred,
free text, empty matches, explicit exact-model terms and a fake model switch.
Relative-day query text stays literal and a later matching image joins the
search. Apply over an old in-flight page, pending debounce, sheet and bin
retains the chosen filter and picks, with one gallery/Runs reload and no image
mutation or generation request. Fixture status polls are drained after direct
synthetic generation before counting Apply requests, so an ordinary other-tab
image catch-up is not mistaken for a second filter transition.

Geometry checks pass at **1400x900, 600x900, 1400x650 and 2048x1040** with
Gallery beside. Narrow/short/wide screenshots were inspected. The new footer
opener initially squeezed wide-beside controls and the selected count; tighter
component spacing fixes that without clipping or adding a header row. Existing
`smoke-gallery`, `smoke-select`, `smoke-sheet` and `smoke-bin` regressions pass,
including the selection's complete outside-view warning. The select/bin smoke
order assertions now include the new footer opener.

This completes the planned local CPU/browser acceptance. It demonstrates
storage and interaction behavior against scratch data; no real gallery,
weights, card fit, image quality, speed or usability improvement was measured.


### Follow-up: uncertain saved writes, 2026-10-28

The [dated recovery review](review-2026-10-28-saved-search-recovery.md) owns
`UI#24`'s reproduced lost-reply feedback and bounded reconciliation repair.
It preserves the existing normalized-name/revision contract and explicit
Apply semantics. CPU/browser qualification is recorded there rather than
inferred from the earlier successful-write CRUD smoke.
