# Personal prompt snippets

`UI#11`, from [G6 of the 2026-10-24 review](review-2026-10-24.md#g6-own-prompt-snippets).

## Scope and behavior

**Examples > Personal snippets…** adds no line, width or height to the
settings caption. It follows the authors' examples and remains available when
there are none. Its native dialog manages named pieces
of text. Each snippet names its destination, Prompt or Negative prompt. The
dialog offers Append (the default) and Prepend; it never replaces the box.
Insertion joins nonempty text with one newline and preserves the saved text
verbatim. It writes through `writeBox`, so one Ctrl+Z restores the box, and
notifies the existing preview, prompt analysis, tags and panel persistence.
A greyed or unsupported negative box refuses insertion with its reason.

Snippets are shared by tabs and checkpoint selections, stored server-side in
`local-diffusion.snippets.json` beside the database and saved loader choices.
They are independent of gallery rows, bundles, model loading and the GPU gate.
The browser reads them on opening the dialog, not at page startup. Nothing is
saved to localStorage. The file is ignored by Git because it holds own prompts.

## Persistence and API

- `GET /api/snippets` returns the collection and its limits.
- `POST /api/snippets` creates one item; `PUT /api/snippets/{id}` updates it;
  `DELETE /api/snippets/{id}?revision=N` removes it. Each item has a random
  server-owned id, name, exact text, target and positive revision.
- An update or delete supplies the last read revision. A stale version is a
  409, leaving the newer item intact; the page asks to reload the list before
  discarding its unsaved editor. Atomic read/modify/write runs under a lock
  independent of the gallery, including across processes.
- Cap the collection at 100 items, names at 80 characters, text at 16,000
  characters and the serialized file at 8 MiB. Refuse unknown fields, targets,
  duplicate ids and malformed/newer files. Never replace an unreadable file
  with an empty collection. Report storage errors in the dialog.
- Writes use a same-directory temporary file, flush/fsync and atomic replace;
  failed writes leave the previous collection and clean their temporary file.

## Dialog and integration

Keep declarations in `web/prompt-snippets.js`, event wiring called explicitly
from `app.js` through `initSnippets()`. Use textContent for saved text and list
labels. The named native dialog contains a list, Name/Text/Target editor,
New/Save/Delete controls and Append/Prepend insertion. It stays open on a
failed request, retaining the editor. Close and native Escape restore the
connected opener without scrolling. The global dispatcher does not generate
under the dialog. Close the dialog before `rewriteBox`, which focuses without
scrolling before `writeBox` performs the browser's undo-aware insertion. The
native modal's inert page must not defeat that write.

## Verification

Unit/API checks cover CRUD, persistence isolation, limits, Unicode and exact
multiline text, malformed-file preservation, atomic failure, stale edits and
deletes, concurrent updates and cross-origin refusal. No model is loaded.
The browser smoke uses a scratch fake server to create/edit/select/delete,
reload the page, append/prepend with Ctrl+Z, refuse a greyed negative, preserve
panel height and contain the dialog at a narrow viewport. Run checks in the
root agent's serial verification window.

## Status

Implemented 2026-10-25 and checked 2026-10-26. The full integrated CPU suite
passes (3,721 passed, 5 skipped), and the isolated snippet/docs-map/smoke-runner
group passes all 35 checks. Ruff and changed-script syntax checks pass.

`smoke-snippets`, `smoke-animagine`, `smoke-ernie` and `smoke-inpaint` pass
serially on Chrome and scratch fake servers. Snippet coverage includes CRUD,
server persistence, concurrent stale edits, no-author-examples access, native
Undo for Append and Prepend, held negative insertion, modal keys and focus,
caption geometry beside ordinary/Tags/Expand controls at both 1400x900 and
400x650, and insertion into an offscreen prompt without page scroll.

A separate caption button added a row on Tags/Expand at 1400 px, so the final
entry is **Examples > Personal snippets…**, after the authors' prompts.
Author-example assertions identify those items explicitly. The native Undo
check also exposed a harness gap: Ctrl+Z now carries its physical Z key code.
Insertion uses the existing `rewriteBox` wrapper after closing the modal;
the shared write helpers are unchanged. No real model was loaded or downloaded.
