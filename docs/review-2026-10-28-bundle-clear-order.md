# Review and plan: Bundle & clear ordering, 2026-10-28

**Status:** closed 2026-10-29. `GAL#12`'s actual complete-clear/late-Restore
defect is reproduced and repaired; the focused fixture passes after a fixture
deadlock fix, and an independent review's draining finding is fixed with its own
negative control (see [final qualification](#final-qualification-2026-10-29)).
This follows
[broad operations](review-2026-10-28-broad-bin-order.md).

## Ranked next work

1. Coordinate Bundle & clear admission, worker lifetime and terminal UI
   application with exact-id and broad lifecycle owners. Its POST starts an
   independent server worker; a request-finally hold would end too soon.
2. Treat unanswered/malformed/replaced receipts conservatively with a sticky
   unresolved hold and explicit inspection/manual-reload guidance. A later
   arbitrary status GET cannot identify an unanswered POST. No auto retry.
3. Durable request correlation, CAS/history/idempotency and automatic recovery
   remain a wider protocol option, not needed to stop unsafe continuation on
   this page. Other clients and external changes retain their stated limits.

## Bounded design and source evidence

Before this repair, Bundle & clear bypassed `binWriteIds`/`binBroadOwner`; server locking
cannot order an earlier Restore's later cache application. Reproduce on a
new owned fake gallery, holding an actual Restore reply while a native
count-confirmed clear finishes. Independently inspect actual committed
deleted ids/storage versus later page cache/feedback. A diagnostic pass proves
the defect. Clear reaches the ENTIRE fake gallery, including its two planted
other-checkpoint rows: explicitly own that complete snapshot rather than
reusing the previous fixture's promise to preserve those rows.

Use one small clear-specific page record beside existing bundle state;
never overwrite `binBroadOwner`. Local clear admission refuses raw exact/
broad owners and claims synchronously before password clearing/POST. Exact/
broad reasons consult the clear reason first, with attribute-only full-control
refresh. Preserve existing count/password payloads and all no-clear backup,
import, settings, model and generation behavior. During an outstanding clear,
hold new bundle starts/settings reservations so last-status evidence cannot
be silently replaced. Read-only status/count/list inspection may continue.

Keep `bundleHold()` unchanged: terminal `refreshBundlePreview` intentionally
returns early under it. Add clear-specific action admission and poll cadence
instead. Valid local receipts bind a nonempty export id and running/done/
failed state. Successful terminal export fields must match those actually
consumed (including committed `deleted_ids`); valid partial cleanup with
`cleared:false` remains valid. Failed-before-archive may have no result.

An observed running export immediately holds NEW lifecycle writes while
existing raw exact/broad owners drain normally. A matching/new relevant
terminal export waits for BOTH owner types before `forgetImages` and awaited
gallery/run/bin/dialog refresh. Do not set `bundleHandled` early. A keyed
settling phase prevents concurrent POST/poll receivers applying the same id
twice. Recheck identity/phase after awaits before handled/release. First-page
already-terminal history remains past; closing the dialog never releases a
live clear or stops polling.

Only direct POST-await 409/422 before any accepted receipt releases that
local attempt. Keep that catch separate from UI application errors. Statusless,
5xx, malformed success, lost/replaced tracked evidence or failed application
remain unresolved. Neither a pending-poll short return nor a pre-POST poll
discarded by request epoch is new evidence. Read-only status inspection can
show server facts but cannot silently clear the unresolved latch or repeat a
mutation. Guidance must survive poll/open/close, say precisely that Bin,
Restore and delete actions stay held, and preserve unsent drafts. Never
automatically reload/navigate; manual reload discards page caches and does
not cancel the server worker or prove the request never ran.

## Qualification

Use the ordinary single-worker Edge harness and a complete owned scratch
gallery. Prove both dispatch orders, POST202/closed-dialog worker lifetime,
observed running/terminal adoption, raw exact+broad draining, held final
refresh and same-id duplicate-receiver exclusion. Include known refusal,
statusless/5xx/malformed receipts, idle/different/replaced status, transient
poll failure and first-page past history. Preserve existing count/password
and ordinary backup/import behavior without accessing a real vault or
activating automatic backups/services. Distinguish actual verified archive/
committed deletion receipts from injected predispatch/transport probes.
Run affected existing smokes, classic/docs/lint, independent source/fixture
review and appropriate visual checks before a local commit. No GPU or user
gallery operation is authorized by a fake fixture.


## Credential-cold preflight and actual baseline

Before browser work, source inspection and independent review traced the
automatic-backup panel GET to `BackupService.status`, which copies cached
metadata. Fresh missing DB-scoped settings have revision `missing`, disabled
state, no gallery identity or credential reference; startup returns before
coordinator/provider construction. The fake sets DB/output/bundle paths under
its new `--data` before configuration import. This exact cold GET performs no
credential API call. Manual password bundle crypto uses the supplied password,
not Windows Credential Manager; the existing manual Bundle smoke has no
automatic-backup activation/settings/reload write.

This is NOT global credential isolation for the fake. A reused directory with
enabled encrypted settings could reach the real worker/provider at startup.
The runner allocates genuinely NEW scratch data before each fake startup.
The 63-step baseline diagnostic fatally checked real status
missing/disabled/not_configured. The focused final fixture requires the same
cold state and guards settings/activation writes before workflow mutations;
its unknown execution outcome is recorded below. A client
stub installed after navigation would not protect reused startup; none was
used or needed. No credential availability/recovery/worker qualification is
claimed and no real vault, service or user gallery was activated.

The **63-step baseline diagnostic passed in 15 seconds** (8 browser seconds)
on 2026-10-28, around 16:18 UTC. It explicitly owned the complete four-row
fake gallery: the planted other-checkpoint pair plus two newly seeded rows.
Native Bin A (#3) committed, then its native Restore committed with the reply
held. Native Bundle & clear/count 4 completed actual archive verification and
clear before that old reply applied. The server reported a done export,
`cleared:true`, deleted ids `[1,2,3,4]`; all four rows and PNGs independently
returned 404, all-model/current/bin counts were 0, and A was locally deleted.
Releasing old Restore removed A's deleted marker and announced `Restored #3.`
despite the actual 404. Other three deleted markers remained. The old epoch
kept all displayed lists empty: no card resurrection is claimed.

The reported archive was present over HTTP with 110622 bytes. This proves
server-reported verification plus actual archive presence, not independent
client ZIP/restore verification. Exactly one seed job and three workflow
writes (Bin, Restore, Clear) were permitted; blocked/error arrays were empty,
and all four observed automatic-backup status reads stayed cold/disabled.
This diagnostic pass proves the defect, not the repair. Artifacts:
`auradream-smoke-artifacts-ixtqnh3_/smoke-bundle-clear-order-aj10fu6v`.

## Applied source checkpoint

A separate clear record now owns admission, known worker/status lifetime,
raw exact+broad drain, keyed terminal application and unresolved outcomes.
Only the direct pre-receipt POST 409/422 catch releases an attempt. Malformed
or unacknowledged/replaced evidence and application exceptions stay held with
authoritative inspection/manual-reload guidance. No automatic navigation,
draft discard or retry is added. Other-tab/missed-history and durable receipt
limits remain above. Successful done metadata is checked before consumption;
valid partial cleanup and nullable failed-before-archive results are retained.

Independent review caught and removed an unnecessary-await race on ordinary
status application and guarded late old-attempt sending cleanup by request
epoch. No-clear backup/import behavior keeps its existing synchronous path.
The final source replay, Node individual/shared scope and two independent
reviews were clear. That source checkpoint passed 35 classic/shared-scope/docs checks
and no-cache Ruff. Existing manual Bundle, 420-step exact ordering and
352-step broad ordering smokes passed sequentially in 94 seconds (32, 27 and
19 browser seconds); artifacts `auradream-smoke-artifacts-unnjhtsn`.
Focused clear-lifecycle regression and visual evidence remain pending here.

## Interrupted final qualification and access boundary

The reviewed final fixture has 465 steps, 396 JavaScript checks, 166 fatal
prerequisites, two screenshots and eight page lifetimes. Before dispatch,
independent review strengthened same-origin checks, one-use exact mutation
approvals and refusal accounting, including fatal zero-counter gates before
all seven navigations. Its SHA256 is
`3B596A3459AC939DB85E508E4CF4834D268E758FE5E7C748149988925AAE49E9`.

The ordinary single-worker run started on 2026-10-28 with process handle
76437, but a desktop disconnect lost that handle. After reconnection it was
unknown. The known fake endpoint on 7943 actively refused a connection, so no
replacement job was started. The exact run-root summary was absent, then
enumeration of `auradream-smoke-artifacts-_pbyyvzf` was denied. Inspection of
that directory and its descendants is stopped; no alternate read, copied
artifact or regenerated screenshot was used to evade the denial.

The final run's result is **unknown**. The baseline and three existing
regressions above remain valid evidence, but they do not complete this focused
fixture or its visual acceptance. `GAL#12` stays OPEN and its source remains
uncommitted while independent source work continues. No real gallery, vault,
model or service was involved. This filesystem denial is separate from the
historical browser launch boundary and the earlier snippet screenshot denial.

## Later independent fake-tool protection

The credential-cold preflight above records the earlier baseline conditions,
when the fake had no global factory refusal. The later independent
[DEV11 repair](review-2026-10-28-fake-credentials.md) now refuses the credential
factory before app startup, with source/stub checks. It does not retry or
qualify this interrupted browser run, its artifacts or visual acceptance;
`GAL#12` remains OPEN with unknown final outcome and uncommitted source.

## Final qualification (2026-10-29)

The earlier interrupted run's unknown outcome is explained by the fixture
itself. Step 89's `eval` ended with
`window.__clReceive=receiveBundleStatus(__clEnd);`, so its completion value
was that promise. `browser.mjs` evaluates with `awaitPromise: true`, and the
promise waits on the reset gate that only step 90 releases. The step therefore
deadlocked until the driver's 300-second limit, which reproduced on a rerun.
The step now ends with `true;`; no other eval ends with a gated promise.

An independent review then found an ordering defect in the repair:
`receiveBundleClearStatus` ran its identity check before its
settling/draining guard. A clear whose terminal result had arrived while an
earlier Restore or Empty bin still held the bin writes stayed `draining`.
The next 750 ms poll could return another client's operation, because the
server keeps only the latest status. That poll then marked the received clear
result unresolved, so its deleted ids were never applied and every clear-held
action stayed held until a reload. The owner now keeps the received terminal
status (`owner.terminal`), and a draining owner applies that status instead of
the newer poll. A `settling` owner returns before the identity check. Three
new fixture steps inject that other-operation poll while draining. With only
the `draining` line removed, the fixture fails fatally at step 91; with it,
the fixture passes (22 s). The review's settling half was largely refuted:
`bundlePollPending` keeps a second poll out during settling. The early return
is kept as defence.

The focused fixture passes on a fresh fake. So do the existing `smoke-bundles`,
`smoke-bin` and related regressions, serially, on the final source (results in
the final run below). The two new screenshots show the Bundles dialog at
1400 and 400 px wide with no clipping or overlap. The held unconfirmed message
sits below their visible fold, so the fixture's assertions, not the shots,
establish it.
Nothing touched a real gallery, credentials, model or GPU.
