# Review and plan: setup list recovery, 2026-10-28

**Status:** both `UI#23` defects reproduced, repaired and locally qualified.

This follows the [picked-favorites shortlist](review-2026-10-28-picked-favorites.md)
and Matt's authorization to choose bounded useful work autonomously. Existing
card/user gates and explicit feature deferrals stay in their owners.

## Ranked shortlist

| Rank | Evidence from current source | Value, scope and risk |
|---|---|---|
| 1 | `selectSetup` increments the shared `setupsTicket` and clears `setupsPending`, even during `reloadSetups`' collection read. The list is enabled during reads. The late GET then fails its ticket check while `setupsReady` is still false. | A normal selection may strand the dialog after Reload. Reproduce with a real held response and native selection; repair the read/selection boundary while preserving the existing ability to change selection during validation. Selected first. |
| 2 | Reload deliberately preserves the old editor/revision, but `renderSetups` also keeps the same option visually selected. Selecting that sole option may produce no native change event, preventing the explicit revision adoption the dialog requests. | Same small recovery boundary. Reproduce against a single externally revised saved item; clear visual selection when Reload preserves the editor, without silently adopting the new revision. Included if confirmed. |
| 3 | Saved-search unknown write results currently claim the operation was not saved/deleted. | Useful honest feedback, but unique names/item revisions already limit duplicate or stale mutation. Independent later candidate; do not expand this slice. |

The analogous snippet/search owners use separate read/edit lifetimes and/or
clear stale visual selection. Their source does not justify a production
change here. The earlier Unicode smoke assigned values and dispatched change
programmatically, so its pass is not native single-option acceptance.

## Qualification and bounded repair

Use ordinary Windows Edge through the existing single-worker scratch fake
harness on 7938. Capture actual native `change` events rather than dispatching
one from a test. Let an actual list GET return from the server, hold its
delivery, and choose a cached item with native input. On current source,
record pending/ready/ticket state before and after release. Exercise close
while a read is pending and verify a late reply cannot update a newer dialog.

For the second issue keep exactly one saved setup, revise it through the
scratch API as another tab, and Reload while retaining an unsaved name. An
explicit native click on the only option must adopt its newest revision;
Reload itself must keep the old editor/revision until then. Assert the
composer is unchanged and no generation or image mutation is submitted.

Prefer a collection-read selection hold plus clear visual selection when
Reload preserves the editor. Keep validation selection cancellation, stale
write protection, uncertain-write fresh-read/explicit-choice recovery and
ordinary New behavior. Add no generic request framework or backend/storage
change. If native reproduction contradicts the hypothesis, document the
result and revise or drop it before implementing.

Run the focused regression, existing setups/write-recovery/Unicode smokes
sequentially, classic-script/docs-map checks and lint. Inspect any changed
states visually if there are layout changes. Claims remain limited to fake
CPU/browser work; no actual model, GPU, credentials or real library is used.


## Native baseline evidence

The diagnostic **62-step** baseline ran against unpatched source after
`ca25874`, on a fresh fake with an explicitly empty saved-setup collection.
All earlier fixture and intended defect assertions passed:

- A trusted native ArrowDown selection changed Alpha to Beta while the real
  GET snapshot was held. The read ticket was 4; selection made it 5 and
  cleared pending while readiness remained false. After releasing the actual
  response, Beta still had `ready=false` and Save requested another Reload.
- With one saved Beta, an external PUT advanced the list to revision 2.
  Reload preserved the dirty name and old revision 1 as intended, but also
  kept native option 0 selected. A trusted option click and native Up/Down
  produced no change event: selected revision stayed 1, list revision was 2,
  and the unsaved name stayed intact. This proves the requested explicit
  selection could not adopt that sole item through those normal gestures.

The diagnostic then tried a possible workaround (New followed by ArrowDown).
Its last two expectations failed because that gesture emitted no native
change on the sole option. That workaround is unqualified; it is not counted
as a regression pass or as evidence against the earlier reproduced defects.
The final regression will use an actual option click after deselection and
assert the resulting trusted change/revision rather than assuming arrow
behavior. Composer, no-generation, page-error and close/focus checks passed.
The baseline returned exit 1 for those two workaround expectations; artifacts
are under `auradream-smoke-artifacts-3w_ff974` in the task workspace.

The proposed repair also preserves the current visible selected value when
the list is rerendered. Otherwise Show other checkpoints could undo Reload's
deliberate deselection by reselecting the retained editor's id. A native
view-toggle/reselection check will cover that necessary part of the same
explicit-adoption contract. No new revision is adopted automatically.


## Final result and acceptance

The small setup-only repair holds the list during collection reads and guards
`selectSetup` before it can invalidate that read. Name and checkpoint-view
editing retain their existing behavior; validation keeps list selection
enabled. Successful Reload clears the visible selection while preserving
the old editor/revision. `renderSetups` preserves that visible selected value
or deliberate blank across view/status rerenders. Save still selects its
confirmed result explicitly; Delete still starts New. No API, backend,
storage, composer or request framework changes were needed.

The final **86-step** native regression passes. It attempts selection during
a held real GET and proves the same GET restores readiness without a second
Reload; native selection during held validation still supersedes the older
check; close/reopen rejects the older GET; a dirty name/old revision survives
sole-item Reload until an actual trusted option click adopts the fresh
revision. View toggles retain both deliberate blank and later explicit
selection. Composer, no-generation, page-error and focus-return checks pass.

The final regression plus existing generation-setups, library-write-recovery
and Unicode saved-name smokes all passed sequentially: **4 of 4 in 84 seconds**,
under `auradream-smoke-artifacts-192u2bi_`. This includes known stale writes,
uncertain-write fresh-read/explicit-choice recovery, Restore/Undo and the
existing responsive geometry checks. A reviewer then requested the final
fixture's explicit empty-collection precondition before its owned POSTs;
that guard was added and the focused smoke passed again, with artifacts in
`auradream-smoke-artifacts-ic1bv8m5/smoke-setup-list-recovery-driqlrfy`.
Production source did not change between those runs.

Classic-script/docs-map checks pass **35 tests**; Ruff and diff checks pass.
The first static run caught the not-yet-added smoke description; adding its
required map resolved that documentation failure. Independent review traced
all render callers, read/validation tickets and close/reopen ownership and
found no blocker. Styles/control positions did not change. These are scratch
CPU/browser interaction results, not user/device, model/card or release
acceptance. No real gallery, credentials, weights or GPU work was used.
