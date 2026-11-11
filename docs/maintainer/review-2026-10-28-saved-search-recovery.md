# Review and plan: saved-search write recovery, 2026-10-28

**Status:** `UI#24` reproduced, repaired and locally qualified.
This continues Matt's authorized autonomous source work after the earlier
[library](review-2026-10-28-library-recovery.md), [gallery](review-2026-10-28-picked-favorites.md)
and [setup-read](review-2026-10-28-setup-list-recovery.md) slices. The owner audit
keeps hardware, user decisions and actual denied operations separate.

## Ranked shortlist

| Rank | Current evidence | User value, scope and risk |
|---|---|---|
| 1 | `saved-searches.js` Save/Delete catches say "was not saved/deleted" after all errors. `state.js` can lose a successful reply or fail reading its body after a write committed; only known 404/409 currently hold readiness. | Honest outcome feedback and deliberate recovery for an existing workflow. Backend normalized-name uniqueness and item revisions already prevent duplicate/stale writes, so this is a low-severity UI correctness fix. One file, no storage/API change. Selected. |
| 2 | Single and batch favorite requests accept opposite values concurrently; each response applies its captured value without ownership across overlapping ids. | Starring is reported Often. Investigate actual server/write/response ordering before choosing a fix; a response token alone cannot establish final server state. Wider scope/risk than rank 1. |
| 3 | The scratch smoke runner's final summary can be missing after a desktop disconnect even when earlier per-smoke receipts passed. | Possible diagnostic hygiene for sustained local work. Existing receipts already allowed safe reconciliation; investigate concrete missing evidence before adding a new runner framework. Lower product priority. |

This ranking uses current local source and observed receipts, not competitor
claims. It does not reopen explicit feature deferrals or activate hardware,
credentials, backup services or real user-data writes.

## Outcome and recovery contract

An error without an HTTP status, or a server 5xx, cannot establish that the
write failed. Preserve the entire saved-search editor and its old selected
revision, say the result is uncertain, clear any duplicate-replacement offer,
and hold Save/Delete/Update named search. Do not retry automatically.

Use the existing Ready/Pending flags for whether a fresh collection read has
completed, plus one uncertainty boolean. Reading/reopening alone does not
acknowledge an unknown result. A deliberate selection or New after a fresh
successful read does. Selection/New before that read may edit the draft but
must retain the write hold. Known 422 refusals remain retryable. Known
duplicate-name 409 responses retain the backend's exact conflict id, fresh
read and explicit Update named search action; no browser name approximation
or implicit replacement is introduced.

Apply remains an explicit use of the valid editor filter, independent of
whether its saved definition was reconciled. Use current filter remains a
draft edit when idle. Neither acknowledges uncertainty; closing/reopening
after Apply must still require a deliberate saved-item selection or New.
Apply retains gallery picks and queues no image work. Existing query checks,
modal ownership, pending-write close holds and stale list tickets stay intact.

## Qualification

First reproduce an actual scratch POST that committed before its reply was
lost. Read storage independently and log the false current feedback before
the desired recovery assertion fails. Use only the ordinary Windows Edge
harness with a fresh FLUX.1 fake on 7939; explicitly refuse nonempty saved
search storage before fixture writes. No model or real gallery is used.

The final smoke must cover committed POST/PUT/DELETE with lost/5xx replies,
no repeated mutations, retained fields/old revisions, failed recovery GET,
fresh-read plus explicit selection/New, empty-list delete recovery and no
automatic acknowledgement on reopening. Include Use current/Apply while
uncertain with existing picks, actual known 422 and normalized-name 409
replacement, and page errors/composer/focus checks. Run relevant existing
saved-search, Unicode and gallery/selection flows sequentially, plus classic
scope/docs-map and lint checks. Review the one-file state change independently
before committing locally. These establish scratch interaction behavior,
not device, card, runtime-quality or release acceptance.


## Baseline reproduction

The **17-step** baseline ran against unpatched source after `08606b7` and
failed only its final desired-recovery assertion. The actual POST committed
exactly one definition (name `Recovery original`, revision 1, exact retained
query). After its successful reply was lost, the UI said `Search was not
saved`, had `ready=true`, and Save's `aria-disabled` was false. Independent
storage proof, retained editor, selected gallery pick and request count all
passed before that assertion. This demonstrates misleading outcome/readiness,
not a bypass of backend name/revision protection. Artifacts are under
`auradream-smoke-artifacts-6ppun75f` in this task's workspace.

Independent source review cleared the one-boolean proposal. The implementation
also keeps recovery instructions visible after Use current filter; that
successful draft edit cannot imply that an uncertain saved write is resolved.


## Final result and acceptance

Saved searches now report an unknown write honestly and preserve the entire
editor/old revision. One uncertainty boolean and the existing Ready/Pending
flags hold Save/Delete/Update named search until a fresh successful read and
explicit selection/New. Automatic opening/new-item helpers cannot acknowledge
it. Use current retains recovery instructions; Apply still uses the valid
editor filter and retains picks without resolving the saved-write result.
Backend uniqueness, revisions, APIs and formats are unchanged. The usage guide
now explains the existing named-filter workflow and this recovery path.

The final **134-step** smoke passes. Actual POST/PUT/DELETE writes commit before
their successful replies are lost or replaced by 503; no held retry writes
are sent. Failed recovery reads, clean/empty-list reopening, native item
selection, old revisions and deliberate New all behave as required. A real
GET is held after its server snapshot: selection/New/typing before delivery
retain uncertainty and the read ticket; release preserves the newer draft,
and only explicit post-read selection acknowledges it. Actual backend 422 and
normalized-name 409 plus explicit revisioned replacement pass. Apply/Use
current retain the chosen filter, pick and composer; page errors stay empty.

The existing **209-step** saved-search smoke also passes. Its pre-commit
synthetic failure formerly had no status while expecting immediate retry;
that now correctly enters uncertainty. The fixture now identifies this known
validation refusal as 422, retaining every editor/stale-write assertion. The
initial new smoke passed its recovery behavior but its final route guard
counted the fixture prompt's read-only encoder measurement as a mutation.
Fresh diagnostics show exactly one `POST /api/prompt/measure`; the final guard
allows only that exact method/path outside library writes and still rejects
image/model mutations. No production workaround was added for either fixture.

Both final smokes passed sequentially (**2 of 2 in 45 seconds**) under
`auradream-smoke-artifacts-jku71daa`. Existing selection and Unicode smokes
already passed on the same production source in
`auradream-smoke-artifacts-w0rb5c5f`; those completed checks were retained rather
than repeated. This qualifies all four affected flows. Three final uncertainty
screenshots were inspected at 1400x900, 400x650 and 1400x600, with contained
feedback and reachable stationary actions. No styles/control positions changed.

Classic-scope/docs-map checks pass **35 tests**; Ruff and diff checks pass.
Independent source/fixture reviews found no blocker. No real image/model
writes, gallery activation, credentials or GPU work were performed. These
results establish scratch interaction behavior and do not complete the older
artifact/download limitations, user decisions or runtime/card acceptance.
