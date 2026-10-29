# Review: close-watcher holds in the saved-library dialogs

**Status:** `UI#35` built and browser-checked on 2026-10-30; complete.
Recorded 2026-10-29 from the TODO audit of that date. All three stages are
done: the six fixes, their smoke steps and pytest scenarios, and the
documents. The evidence is in
[Qualification, 2026-10-30](#qualification-2026-10-30). The file:line
references below describe the source before the fixes.

## Background: the second Escape

A native `<dialog>` held by `event.preventDefault()` in its `cancel` handler
is not held against every Escape. Chrome's close-watcher rule lets a second
Escape with no user activation in between close the dialog anyway. The
`cancel` event is then not cancelable, and `close` fires as usual. Three
places already accept this:

- `web/setup-import.js:387-398`. The import dialog's `close` handler leaves
  a pending commit's ticket and busy state alone, so the commit's reply
  still lands. Stage 3's review fixes are in `docs/plan-setup-portability.md`
  under "Review fixes, 2026-10-29". There, the real second Escape closed the
  dialog in the Windows run.
- `web/refine.js:903-906`. An accepted Refine pass takes a forced close as
  the run it is.
- `tools/smoke-setup-portability.json:205-232`. The smoke presses Escape
  twice during a held commit. If the browser still holds the dialog, it
  falls back to `dialog.close()`.

Two places do not accept it yet. The snippet import dialog still has the
three defects that stage 3 fixed in setup-import.js. The parent Saved setups
and Snippets dialogs drop their write hold on any close, so a forced close
during a write lets a second Save create a duplicate.

`docs/design/preferences.md` has no feature id for personal snippets, saved
setups or saved searches, so this row is ranked as a defect only. The
duplicates in (b) are the outcome that `UI#21`
(`docs/review-2026-10-28-library-recovery.md`) set out to prevent for lost
Create replies.

## (a) Snippet import: copies of the three setup-import defects

### A1. A second Escape during a commit drops the reply

`commitSnippetImport` (`web/snippet-import.js:236-264`) takes a ticket and
sets `snippetImportBusy = 'commit'` and the parent's
`snippetsPending = 'write'` (line 241). The `cancel` handler (304-306) holds
the first Escape. The `close` handler (307-312) always runs
`++snippetImportTicket; snippetImportBusy = null`.

Failure sequence:

1. Import a file, Preview, then press Import. The commit is in flight.
2. Press Escape (held: "Wait for the import to finish."), then Escape again
   with nothing pressed in between. The import dialog closes.
3. The `close` handler retires the commit's ticket. When the reply arrives,
   line 248 (success) or line 254 (failure) returns early, and the `finally`
   at 262 skips its cleanup, so `snippetsPending` stays `'write'`.
4. The Snippets dialog is left frozen. Save, New and Close are held with
   "Wait for the snippet to finish saving." (`web/prompt-snippets.js:42,
   77-78`), Delete and Reload with "Wait for the current snippets request."
   (36, 74), and the list and fields are disabled (79-81). Its own
   `cancel` handler (328-330) holds Escape too. A successful import never
   refreshes the list, and an uncertain or refused one is never reported. The only way out is a forced close of the parent,
   which is defect B1.

The fix is setup-import.js:390-398: the `close` handler leaves a pending
commit's ticket and busy state alone. Also port setup-import.js:331 and
338-339. When the import dialog is already shut, a refused or uncertain
result is reported in the Snippets status (`snippetSay`), because
`snippetImportSay` (259-260) writes into a closed dialog.

### A2. A previewed Add-as-copy row loses Replace and gets a 422

The server matches each row under its effective name, which is the copy name
for an Add-as-copy choice (`backend/snippet_portability.py:180-183, 194`).
`renderSnippetImport` stores the reply's rows as `snippetImportRows` (line
91). It draws the heading (102), the saved matches (112-119), Add/Replace
availability (127-131) and the replacement targets (140-143) from those
rows. `snippetImportChoiceReason` (41-45) checks against the same rows.

Failure sequence:

1. Saved snippets contain "Harbour". Import a file whose item #1 is
   "Harbour". The default preview offers Skip / Add as copy / Replace.
2. Choose Add as copy, name it "Harbour 2", then press Preview.
3. The redrawn row is headed "#1 Harbour 2" with no saved match. Add is
   offered as plain "Add", and Replace is disabled.
4. Choose Add. The client check passes, because the stale row has no
   matches. Preview then gets the server's 422, "A colliding snippet needs
   Skip, Add as copy or an explicit replacement."
   (`snippet_portability.py:200-201`). Replace stays unreachable until the
   row goes back to Skip and is previewed again.

The fix is the `setupImportIncoming` map (setup-import.js:13-17, 63-65, 136,
145). For each file index, keep the latest row previewed under the file
item's own name. Draw the heading, the saved matches, Add/Replace, the
replacement targets and the client hold from that row. Only the
incoming-collision note and the counts come from the copy preview. The map
is reset where `openSnippetImport` resets the rows (line 205).

### A3. A refreshing Preview loses edits

After a 409 or an uncertain commit, `snippetImportNeedsReload` is true, and
Preview first rereads the list and the default rows (175-186). Line 184,
`renderSnippetImport(defaults, true)`, builds new controls that start
enabled. `syncSnippetPortability` is the only code that disables row
controls (line 68), and it is not called after line 184. So the rows stay
editable while `snippetImportBusy === 'preview'`. Line 187 then sends the
final preview, and line 191 renders its reply with `keepChoices` false.

Failure sequence:

1. Make a commit get a 409, or lose its reply. Then press Preview.
2. While the final preview is in flight, change row #1 from Replace to Skip.
3. The reply arrives. Line 191 replaces `snippetImportChoices` with the
   rows the server echoed, which still say Replace. The row control reverts
   on screen, and `snippetImportShown` is set, so Import is enabled for the
   choice the user just changed away from.

The fix is setup-import.js:242, which holds the redrawn rows again at once,
and 245 and 251, which keep the request body and ignore a reply whose body
no longer matches the current choices. That case shows "Choices changed.
Preview again before importing."

## (b) Forced close clears the parent write hold

The same second Escape can close the parent dialog during an ordinary write.
Both `close` handlers then clear `*Pending` without checking for a write.
A write's reply carries no ticket, so it still lands, but nothing holds a
second write in the meantime.

### B1. Snippets: a second Save duplicates the snippet

`snippetsClosed` (`web/prompt-snippets.js:265-275`) sets
`snippetsPending = null` on any close. The `cancel` guard (328-330) holds
only the first Escape.

Failure sequence:

1. Choose New, type a name and text, and press Save. `saveSnippet` (192-226)
   sets `snippetsPending = 'write'` and POSTs `/api/snippets`. Its reply is
   slow.
2. Press Escape twice. The dialog closes and the hold is cleared.
3. Reopen Examples > Personal snippets…. `readSnippets({ opening: true })`
   passes its check at line 142. The editor is dirty, so it is kept
   (162, 167), and Save is enabled once the list arrives (41-52).
4. Press Save. `snippetsSelected` is still null, so this is a second POST,
   and the server appends a second snippet with the same name. The backend
   allows duplicate names; `backend/snippets.py` refuses only duplicate ids
   (line 51).

If the first reply lands before step 4, `selectSnippet` adopts the saved
item and step 4 becomes a PUT. The duplicate needs the first reply to be
slower than a close and a reopen. A gated reply in the smoke makes that
deterministic.

### B2. Saved setups: a second Save current duplicates the setup

The `close` handler (`web/generation-presets.js:511-518`) runs
`++setupsTicket; setupsPending = null`. The sequence is B1's, with Save
current: `saveSetup('new')` (316-346) always POSTs, and its hold (line 129)
is clear again after the reopening `reloadSetups` (277-307). Duplicate setup
names are legal by design (`docs/plan-setup-portability.md`, "Identity and
merge behavior"), so the server stores both.

### B3. Saved setups: a reply that lands while closed leaves the dialog stuck

After a forced close, the slow first reply still runs `saveSetup`'s success
path (332-337). That path calls `selectSetup(saved.id, false)` (238-260),
which, with no restore hold, calls `checkSetup` (197-236).
`checkSetup` sets `setupsPending = 'validate'` (203). Its reply returns at
line 210 or 228, because the dialog is closed, and `setupsPending` is never
cleared.

Failure sequence: B2's steps 1-2. Wait for the reply, then reopen Saved
setups…. `reloadSetups` refuses at line 278 with "Wait for the saved setup
request.", so the list is never read again. Reload, Save current, New and
Models are held with the same reason (lines 120, 128-136). The dialog stays
stuck until it is closed and opened again.

### Fix for (b)

- A `close` handler retires reads and validations as it does now, but
  leaves a pending `'write'` in place. Only the write's own reply clears it.
  This applies to `snippetsClosed` and to the Saved setups `close` handler.
  Write replies already return the complete list (`result.items`), so the
  list is current when the reply lands.
- A reopened dialog opens with writes still held, using the existing
  reasons ("Wait for the snippet to finish saving." / "Wait for the saved
  setup request."). Its opening read is refused, as
  `readSnippets` (142) and `reloadSetups` (278) refuse it today. The write's
  reply updates the list in place. No new control or text is added.
- A write reply that lands while the dialog is closed updates the list and
  the selection, but it does not start a validation. `checkSetup` (or
  `selectSetup`'s call to it) does nothing while `els.setupsDialog` is
  closed. Restore revalidates on the next press anyway.
- The import commits (`setupsPending` / `snippetsPending = 'write'`, set by
  `commitSetupImport` and `commitSnippetImport`) keep the same hold through a
  forced close of the parent, which is consistent with A1.

### Checked and dropped: Saved searches

`savedSearchesClosed` (`web/saved-searches.js:301-308`) also clears
`savedSearchPending` on a forced close, and its `cancel` handler (314-317)
only calls `closeSavedSearches`, which refuses during a write (294-299). A
second Save after a reopen does not duplicate anything, though. The server
refuses a second item with the same name under NFC/casefold comparison
(`backend/saved_searches.py:128-132`, 409 with `conflict_id`), and a second
PUT or DELETE fails the revision check. The result is the existing
"That name is already saved" conflict message, which is ordinary behaviour.
No harmful sequence could be traced, so Saved searches is left out of this
row. If (b)'s fix is ever applied there for consistency, it needs its own
row.

## Constraints

- A held button stays `aria-disabled` with its reason as its title, never
  `disabled` (AGENTS.md, Frontend). The existing reasons are reused. Row
  `<input>`/`<select>` controls keep their current `disabled` treatment
  during a request (snippet-import.js:68, setup-import.js:84).
- The page never moves a control under the pointer and gains no height. No
  control, row or note is added to either dialog or to the generation panel
  (AGENTS.md, Frontend; `docs/rules.md`, under-1800 px rule).
- The server decides the merge. Matches, collisions and the 409/422 refusals
  stay in `snippet_portability.plan_merge` and are not changed. The page
  only stops redrawing from a copy-named row.
- An uncertain import or write is never resent. The existing
  `snippetImportCopyConfirm` and `snippetsWriteRecovery` /
  `setupsWriteRecovery` paths stay as they are.
- Top-level bindings stay top-level, and `index.html`'s script order is
  unchanged (the classic-script scope guard in pytest).
- `insertSnippet`'s writes through `rewriteBox` are untouched. Neither
  dialog writes the prompt or negative here.

## Non-goals

- Stopping the second Escape itself, for example by capturing the Escape
  `keydown`. setup-import.js and refine.js accept the forced close, and this
  row keeps that model.
- Changing the backend, the portable formats, duplicate-name legality or
  the merge rules.
- Saved searches (see above), Saved setups' import dialog (already fixed)
  and any other dialog.
- Consolidating snippet-import.js and setup-import.js into one module.

## Stages and completion criteria

### Stage 1: snippet import (A1-A3)

- Before each fix, reproduce its defect on the original source with the new
  smoke step, and record that the step fails.
- Port the three fixes into `web/snippet-import.js`.
- Add steps to `tools/smoke-snippet-portability.json` (port 7934, fresh
  scratch FLUX.1 Krea [dev] fake), using a gated commit as
  smoke-setup-portability does:
  - A1: two `key: Escape` presses during a held commit (falling back to
    `dialog.close()`), then expect `snippetImportBusy === 'commit'` and
    `snippetsPending === 'write'`. Release the reply, then expect the list
    refreshed and the parent hold cleared. A second variant uses a lost
    reply and expects the uncertainty in the Snippets status.
  - A2: a previewed Add-as-copy row still shows its saved match and
    offers Replace, Add stays disabled, and no 422 preview is sent.
  - A3: during a gated refreshing Preview, the redrawn rows are disabled.
    A choice changed by script before the release survives, the status
    reads "Choices changed. Preview again before importing.", and Import
    stays held (`snippetImportShown` stays null).
- Each step fails with only its fix reverted.
- Also pass: `smoke-snippets`, `smoke-library-write-recovery`,
  `smoke-saved-name-unicode`, `uv run --group dev pytest` (scope guard
  included) and `ruff check`.

### Stage 2: parent write holds (B1-B3)

- Before each fix, reproduce its defect on the original source, and record
  that the step fails.
- Change the two parent `close` handlers, and the `checkSetup` call path, as
  described in "Fix for (b)".
- Add steps to `tools/smoke-library-write-recovery.json` (port 7935), which
  already wraps the snippet and setup write calls. For each dialog: gate a
  Create reply, press Escape twice (falling back to `dialog.close()`),
  reopen, and expect Save held with its reason. Release the reply, then
  expect exactly one new item on the server (counted through the API) and
  the editor adopting it. For Saved setups, also expect `setupsPending`
  cleared after the reply, and Reload available after the reopen (B3).
- Each step fails with only its fix reverted.
- Also pass: `smoke-generation-presets`, `smoke-setup-list-recovery`,
  `smoke-setup-portability`, `smoke-snippets`, pytest and ruff.

### Stage 3: documents

- `docs/rules.md` Tests paragraphs for the two smokes name the new coverage.
- `docs/plan-snippet-portability.md` and `docs/plan-setup-portability.md`
  point here.
- This document gets a dated qualification section with the step counts and
  runs.

No stage needs the card, the real gallery or downloads. This is
browser-only UI state, checked on the fake server.

## Qualification, 2026-10-30

Built on the fake server and in headless Chrome on Windows; no card, real
gallery or download was used.

**What changed.**

- `web/snippet-import.js` takes setup-import.js's three fixes. A1: the
  import dialog's `close` handler leaves a pending commit's ticket and busy
  state alone. A refused or uncertain reply that lands after the dialog has
  shut is also reported in the Snippets status. A2: `snippetImportIncoming`
  keeps, per file index, the latest row previewed under the file item's own
  name. The heading, saved matches, Add/Replace, replacement targets and the
  client hold are drawn from it. A3: the rows redrawn by a refreshing Preview
  are held at once. A reply whose request body no longer matches the current
  choices is ignored with "Choices changed. Preview again before importing."
- `web/prompt-snippets.js` (`snippetsClosed`) and the Saved setups `close`
  handler in `web/generation-presets.js` retire reads and validations but
  leave a pending `'write'` in place. `checkSetup` does nothing while Saved
  setups is closed, so a write reply that lands then selects the new item
  without starting a validation (B3).
- When a reopened dialog's opening read is refused, the refusal now calls
  `syncSnippets()` / `syncSetups()` once, so the held buttons show the
  existing reasons. No control, text or height was added.

**Smokes.** `smoke-snippet-portability` grew from 183 steps to 245 (110
expectations, 8 fatal prerequisites). The existing held-commit section now
presses Escape twice and checks the commit and the parent's hold. New
sections at the end cover a forced close with a lost reply (A1), a
previewed Add-as-copy row (A2) and a refreshing Preview gated in flight
(A3). `smoke-library-write-recovery` grew from 221 steps to 288 (125
expectations, 8 fatal). New sections force-close Snippets and then Saved
setups during a gated Create, and reopen each one. Save stays held with its
reason, and exactly one item exists on the server after the reply. A third
section lets a Save current reply land while Saved setups is closed, and
then checks that no validation was sent and that Reload is available after
the reopen (B3). Both smokes passed on Windows in 17 s and 31 s. In both,
the real second Escape closed the dialog, so the `dialog.close()` fallback
was not needed.

**Reproduction.** Taken while Matt was away (2026-10-29), revisit if
wanted: the stages asked for each defect to be reproduced on the original
source before its fix. Instead, every fix was reverted alone, with the other
five in place, and the smoke and pytest were run against that tree. Each
reversal failed its own steps: A1 at the forced-close expectation (and every
later commit check), A2 at the copy-row heading/Replace check, A3 at the
held-rows check and the "Choices changed" reply, B1 and B2 at the reopened
Save hold and the single-POST count, and B3 at `!setupsPending` after the
closed-dialog reply. The new scenarios in
`tests/test_close_watcher_holds.py` (9 scenarios, run in Node `vm` over the
actual modules) failed the same way.

**Also passed:** `smoke-snippets`, `smoke-saved-name-unicode`,
`smoke-generation-presets`, `smoke-setup-list-recovery` and
`smoke-setup-portability`, plus ruff, the pinned type gate and the full
`pytest -n auto`.
