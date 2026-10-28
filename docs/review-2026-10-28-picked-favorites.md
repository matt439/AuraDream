# Review and plan: hidden picked favorites, 2026-10-28

**Status:** `GAL#8` reproduced, repaired and locally qualified.
This is the next bounded improvement under Matt's autonomous-work request.
The remaining older OPEN rows have explicit artifact/download limitations;
the full owner audit is in the [saved-library review](review-2026-10-28-library-recovery.md).
User/card gates and explicitly deferred features retain their owners.

## Ranked shortlist

| Rank | Current source evidence | Value, scope and risk |
|---|---|---|
| 1 | `gallery-actions.js` confirms stars through `markFavorite`, but `recordsWithId` omits the active/stashed selection maps. Hidden picks can be absent from every array it updates. `selectedRecords` then supplies their retained snapshots directly to Compare picked. | Picking across filters (`E12`) and starring (`F5`) are both reported Often in [preferences](design/preferences.md). A narrow confirmed-state propagation fix, with real fake-server/browser reproduction. Selected first; low risk if membership/order and bin ownership stay unchanged. |
| 2 | `generation-presets.js` permits selection during a pending list read; `selectSetup` increments the request ticket while the collection is not ready. Reload also retains an old selected revision while visually keeping its option selected. | Potentially confusing saved-setup recovery and stale controls. Needs separate reproduction and careful native-select/request-lifetime checks. Independent follow-up candidate. |
| 3 | Opposite favorite writes, or Restore followed by another Bin, can return in a different order from their writes. | Broader state-ordering work with overlapping ids and server admission semantics. More risk and scope than rank 1; retain for investigation rather than applying response-token guesses. |

The ranking uses current local source and the recorded tasks. It makes no
competitor claims and does not authorize runtime/GPU trials, credential work,
real gallery mutation or deferred product features.

## Contract and qualification

First reproduce on the ordinary Windows Edge smoke harness with fresh scratch
FLUX.1 fake data. Create three owned synthetic records, then navigate afresh
so the older picked pair is absent from viewer/session/lightbox arrays. Pick
the pair, filter it away, Star picked and inspect storage, retained snapshots
and Compare picked independently. A failing earlier fixture assertion is not
a reproduction. Leave the fake's other-model seed records intact.

If reproduced, extend only `recordsWithId` to include active and stashed pick
values. Both individual and batch confirmed stars already call this helper.
Keep pick membership/order/anchors, API semantics, rendering, data formats and
request ordering unchanged. Repeated aliases may receive the same idempotent
assignment. No new read is required to open Compare picked.

The regression must cover hidden active picks, Unstar, already-starred server
answers and a committed favorite whose response arrives while the original
picks are set aside in Bin. Keep the bin's independent selection unchanged;
restored gallery picks retain order and confirmed stars. Check composer
retention, no exact record reread for comparison and no page errors. Review
the small patch independently, run selection/lightbox/bin compatibility
smokes sequentially, and run classic-scope/docs-map/lint checks. Browser
acceptance here is separate from all earlier denied artifact targets.


## Baseline reproduction

The unpatched **86-step** smoke reached its intended behavior assertions.
The server rows for owned ids 3 and 4 both had `favorite=true`; neither id
was held in viewer/session/lightbox arrays, but the hidden selection records
remained false. Compare picked opened without an exact record reread and
its actual star had `aria-pressed=false`. The later independent Bin phase
likewise received a successful committed favorite reply while the originals
were stashed, yet both stashed flags remained false. Seven expectations
failed across these effects; earlier fixture/isolation checks passed.
Artifacts were reported beneath `auradream-smoke-artifacts-xpn7zyw4`.

During the running diagnostic, reading that new run's
`smoke-picked-favorites-0jgn_tn2/browser.log` was denied, as was a read-only
`Get-CimInstance Win32_Process` query for owned test process status. Neither
action was retried or accessed by another route. The harness then completed
normally and returned the diagnostic output above through its original
command. These denied reads do not qualify visual screenshots and are
separate from the earlier denied portability directory and compiler download.


## Result and qualification

The production change adds active and stashed selection values to the existing
by-id confirmed favorite propagation helper. No request, selected id, order,
anchor or data format changes. The final **86-step** regression passes with
actual stored true/false values, hidden snapshot updates, both comparison
members' pressed stars, already-starred responses and a committed reply
arriving while original picks are stashed. The independent bin pick remains
false and selected; leaving Bin restores the original pair/order. Comparison
uses zero exact-record rereads, and composer text/page-error checks pass.

The disconnect interrupted the first qualification command after selection,
lightbox and bin smokes completed. Their existing `result.json` receipts in
`auradream-smoke-artifacts-edl5a45u` confirm **all three passed**. The fourth
smoke's output showed correct active hidden/already-starred behavior but no
final receipt, so it was not counted as a pass. The original exec session
could not resume; all four known scratch endpoints had closed, and the
documented 300-second browser-answer window elapsed without further output.
Only that unfinished smoke was rerun. Its independent full pass is retained
in `auradream-smoke-artifacts-go3eqf5l/smoke-picked-favorites-gioe96c2`.
No completed compatibility test or commit was repeated.

Classic-scope/docs-map checks passed **35 tests**; Ruff and diff checks pass.
Independent review found no blocker. The active-pair screenshot from the
first qualified behavior and the final stashed-pair screenshot were inspected
at 1400x900: both display the confirmed pressed star and retained comparison
controls. No styles/control positions changed. The earlier denied diagnostic
targets were not retried, and none of this completes older blocked work.
All images and writes were owned fake scratch data; no model, card, real
gallery, compiler or credentials were involved.
