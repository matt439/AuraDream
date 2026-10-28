# Review and plan: Bundle & clear ordering, 2026-10-28

**Status:** `GAL#12` selected for controlled owned-fake reproduction and a
bounded source repair after qualified [broad operations](review-2026-10-28-broad-bin-order.md).

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

Current Bundle & clear bypasses `binWriteIds`/`binBroadOwner`; server locking
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
