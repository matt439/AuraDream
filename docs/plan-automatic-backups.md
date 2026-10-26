# Plan: automatic whole-gallery backups

**Status, 2026-10-26: planning and the pure CPU policy slice are complete;
`GAL#6` remains open for durable ownership, coordinator implementation and
activation decisions.** This expands
[R8](review-2026-10-25.md#r8-automatic-whole-gallery-backup). No live export,
archive deletion, credential access or configuration change was performed.
Automatic backups remain off. Proposed policies below require an explicit
choice before activation; this plan selects no age, retention count or
password policy for Matt.

## Existing path to reuse

`bundles.export_bundle(keep_gallery=True, ids=None)` already exports every
checkpoint and the recycle bin, preserving stars, lineage, masks and the
whole-gallery file inventory, including unindexed image files. It snapshots
SQLite through the backup API, folding in WAL, writes an archive partial,
fsyncs it, performs the existing full verifying read-back, publishes it and
returns `kept=True`, `cleared=False`, `deleted_ids=[]`. It has no typed-count
gate and performs no source cleanup. Automatic backups must call this path,
not copy the database/PNGs independently or call the clearing export.

The HTTP side's `BundleManager.reserve/start/finish` already prevents
conflicting jobs, model switches and complete gallery/image response
streams, retains admission until its file-writing worker ends and waits
for that worker during shutdown. The worker's `gallery_lock.hold` covers
other processes and precedes SQLite's write lock. Reuse those two layers;
do not invent a second maintenance flag or acquire locks in reverse order.

These facts were checked in `api_bundles.py`, `bundles.py`,
`gallery_lock.py`, `storage.bundle_directory`, the app lifespan and
`tests/test_bundles_backup.py`, against the rules' Gallery bundles contract
and design notes' [bundle section](design-notes.md#gallery-bundles-verify-before-clearing-stage-before-merging).

## Activation, age and saved state

Use gallery-scoped, atomically saved backup preferences, independent of
the four model/startup choices. An explicit checkpoint override must not
disable the gallery's protection or accidentally enable it. A dedicated
versioned sidecar beside the database is suitable; reuse the existing
atomic-write discipline without putting unrelated fields in
`StartupPreferences`. Proposed fields are `enabled`, `minimum_age_hours`,
`keep_last`, `protection` and an opaque credential reference where needed.
Validate finite positive age, bounded positive retention count and an
explicit protection policy. Old/missing settings read disabled; malformed
settings show a repairable warning and do no work.

Keep operational receipts separate from user preferences: gallery/owner
identifier, operation UUID, archive basename, successful verification time
in UTC, protection mode, and publication/retention outcome. Persist success
only after the ordinary export's read-back and final publication succeed.
A write-start timestamp, file mtime, parseable filename or `.part` never
counts as a backup. Retain the newest successful receipt across a failed
attempt. A missing archive invalidates that receipt for age purposes; an
unreadable or uncertain archive is reported and never assumed safe.

At each server start, make one due decision, with no recurring timer in
this slice. Proposed due rule: no successful managed copy, or elapsed UTC
time since the newest available successful managed copy at least the saved
age. Clamp backward-clock elapsed time to zero and surface a clock warning;
future timestamps must not cause destructive catch-up work. Recheck
receipts and age inside the cross-process reservation immediately before
export, so simultaneous server starts do not create duplicate backups.
Decide explicitly whether a later manual whole-gallery backup may also
satisfy age; selected/starred bundles never establish whole-gallery cover.

An empty gallery is a separate outcome, not a successful replacement of a
previous nonempty backup. Proposed first slice skips it without retention
or success-age advancement, since a missing/restored database might be the
reason it is empty. Include the decision in the eventual activation UI.

## Startup, preload and shutdown

After the existing database initialization and partial settlement, inspect
backup preferences/receipts off the event loop. If disabled or not due,
retain today's startup and preload behavior. If due, reserve the existing
`BundleManager` synchronously before accepting gallery-changing requests,
then start one owned worker. Serve status and permitted bundle downloads
while it runs. Display its origin as automatic startup backup, its scope
as all models including the recycle bin, and its chosen protection mode.

Proposed sequencing is backup before optional model preload, avoiding two
memory-heavy startup operations together. Schedule preload after this
attempt finishes, whether it succeeds or fails; a failed backup must not
prevent the user from opening the app and repairing the problem. There is
no generation warm-up or model load initiated by the backup itself. The
existing admission check still governs a manual Load or other early API
request; do not hold a generation gate while awaiting the load lock.

If another process holds the gallery reservation, report a deferred startup
attempt and leave the success receipt unchanged. Do not poll forever or
preempt the other process. A bounded later retry or "Run backup now" is an
explicit follow-up policy; the first slice may offer a manual retry through
the existing bundle manager. No failed attempt starts an automatic loop.

On graceful shutdown, cancel only a due check which has not admitted a
worker. Once admitted, call the existing `BundleManager.finish` and keep
its reservation until the thread actually ends. Cancellation of an asyncio
supervisor cannot cancel a file-writing thread. Do not delete its partial
or release admission early. After a hard termination, the next start uses
the established stale-partial settlement under reservation; unfinished
work never resets age or starts retention.

## Password policy: Matt's decision

| Activation choice | Required behavior |
|---|---|
| Leave automatic backup disabled | Current manual password-backed bundles remain available |
| Explicitly allow unencrypted automatic backups | Say plainly that archive contents and prompts are unencrypted, show the destination, write ordinary ZIPs |
| Use a named OS credential-store entry | Retrieve only for the owned worker; stream the existing encrypted `.ldbundle` format and verify it with that secret |

An OS-backed option needs a separately designed credential setup/recovery
flow and dependency review; this plan neither installs `keyring` nor
assumes a working Windows or WSL backend. Require a positively identified,
approved OS backend. Refuse plaintext/null-backend fallbacks, interactive
unlock prompts in the startup worker, missing credentials, locked stores
and unsupported headless environments. Keep the old backup copies and
report the specific repair action. Never silently change protection mode.

Preferences, receipts, command arguments, URLs, logs, status and errors
carry no password. Do not reuse the browser's transient export/import
field or persist its value. Store only an opaque credential reference in
settings, scope that entry to this app/gallery and pass the retrieved
secret directly to the current export. Release the reference after work;
do not promise deterministic memory zeroing of a Python string. Rotation
must leave old copies restorable and explain which credential they need.
An OS-stored secret is not itself a recovery plan for losing this PC:
activation must cover how Matt retains access to encrypted archives.

## Managed archives and retention

Keep all archives under the existing `storage.bundle_directory()` policy,
including its configured external destination, link/junction refusals and
separation from outputs. Introduce explicit automation ownership metadata
and an atomic receipt journal before implementing deletion. Use unique
operation ids and exact archive basenames. Filename prefixes, suffixes,
creation times or "oldest file in bundles" alone never confer ownership.
Manual backups, clearing bundles, imports, unknown files, another gallery's
archives and copies without a validated ownership receipt are ineligible
for automatic deletion.

The implementation stage must choose a durable identity binding between
receipt and verified archive, with tests for a replaced same-name file.
An optional ownership marker in the verified manifest plus its receipt is
one proposal; verify backward-compatible import handling before adopting
it. Do not add a redundant full read-back of a newly created archive just
to establish that marker. Uncertain ownership always retains the file.

Retention runs only after a new successful archive and its durable receipt
exist, under the same whole operation's cross-process reservation. Keep
at least the requested N most recent confirmed automation-owned successes
for this gallery/destination. Never pre-delete a good archive to create
space, prune after a failed/empty/skipped export, or count a failed
publication toward N. Determine candidates before deleting and validate
ownership, regular-file status, containment and current identity at each
unlink; no recursive directory removal. Do not touch a file with a live
download lease; downloads are admitted separately today, so this needs an
explicit short-lived lease/retention exclusion rather than an assumption
that maintenance blocks them.

Retention failures do not erase a successful backup. Record success and
the retained excess copies, explain that cleanup failed and continue
serving the gallery. Crash recovery reconciles pending receipts
conservatively: a published file without a completed receipt is kept
unmanaged until a deliberate verification recovers ownership. It is never
deleted merely because journal publication was interrupted. A damaged or
missing journal disables pruning until repaired; do not reconstruct
deletion authority from names. Destination changes start another managed
set and never prune the old location implicitly.

## Space checks, scope and presentation

Retain export's per-volume `_check_space` for the archive plus database
snapshot/WAL and any retained-job snapshot compaction. Before exposing
automatic operation, account explicitly for ZIP member overhead,
encryption overhead and receipt writes; the existing estimate does not
include all of these. A full or unavailable destination refuses before
writing where possible. Disk races and write/verify/fsync failures still
remove only the worker's own partials and leave prior good copies intact.
An automatic backup on the same physical disk offers rollback, not proof
of protection against loss of that disk; destination choice remains visible.

Status distinguishes disabled, not due, deferred, running, succeeded,
failed and retention warning, with verification time, destination and
records/bytes where known. Do not say "backed up" until verified publish.
Reuse bundle progress phases and the existing non-repeated live status;
automatic origin must not make another tab forget images or show a clear
result. Preserve the current recovery behavior for gallery reads refused
while maintenance ran. Settings and status are gallery-scoped and contain
no prompts, passwords or per-model scope confusion.

## Bounded implementation and checks

**Completed CPU contract slice, 2026-10-26.** `backend/backup_policy.py`
adds a default-off policy requiring explicit positive age, retention and
protection before it can be enabled, plus pure `backup_due` and
`retention_plan` functions. Inputs are supplied verification/ownership
facts and an aware clock, not filenames or a new receipt format. Only
currently verified managed copies for the exact gallery, destination and
protection set cover age or enter retention; the new success, newest N and
download leases are protected. Untrusted ownership, unsuccessful attempts
and backward clocks produce no pruning candidates. Manual/unknown/missing/
replaced copies confer no deletion authority. No filesystem, app lifecycle,
export, credential, settings write or activation is connected.

All 27 focused tests passed with `--noconftest` (no app fixtures/imports),
covering age boundaries/offsets, false trust, missing/changed copies,
empty/skipped/failed attempts, protection/destination changes, retained
counts, leases and clock corrections. Verification times are normalized
to UTC so a repeated wall hour's `fold` cannot reverse chronological
order; contradictory same-scope archive identities are rejected before
filtering copy states. Focused ruff passed and the pinned
Pyright check reports zero errors/warnings. This is a partial stage 1:
saved preferences, durable identity binding/journal, user policy decisions
and all execution/acceptance stages below remain unimplemented.

1. **Policy and journal contracts, CPU only.** Settle the decisions below;
   implement validated default-off preferences and a pure due/retention
   planner with fake clocks and temporary paths. Cover old/malformed
   settings, UTC age boundary, backward clock, missing files, empty gallery,
   ownership/replaced-file rejection, N retention and destination changes.
   No deletion is enabled before durable ownership is established.
2. **Startup coordinator and verified export, scratch galleries only.**
   Reuse `BundleManager`, its shutdown drain and `gallery_lock` in one
   worker. Cover simultaneous starts, manual backup/import/clear, a live
   image stream, queued generation, switch and preload ordering. Inject
   failures at disk-space, write, verify, publish, receipt publication and
   each retention unlink. Assert the live gallery is byte-for-byte
   unchanged, old good archives survive, success age is honest, locks are
   held until thread exit and downloads cannot race deletion.
3. **Protection and UI.** Implement only the policy Matt chooses. For the
   OS option, unit tests use a fake credential provider and integration
   checks require the intended OS backend separately. Verify missing/locked
   credentials produce no plaintext archive or secret output. Extend the
   scratch-fake Bundles smoke for activation, status, held actions, retry,
   reload/other-tab outcomes and retained excess copies. Run focused bundle,
   lock, lifecycle, preferences and docs-map tests plus ruff and the normal
   CPU gate. No model, real gallery or GPU is required.
4. **Acceptance.** Demonstrate two server starts over one synthetic gallery,
   restore the resulting automatic archive into a second scratch gallery
   through normal verified import, and prove all checkpoints, bin, stars,
   lineage and masks survived. Check an existing manual archive stays
   untouched and a deliberate failed successor preserves the previous
   successful copy. Actual OS credential delivery remains unverified until
   exercised on that OS; no fake-provider result closes it.

## Decisions before activation

Matt chooses unattended protection, age and N, destination/recovery policy,
whether manual whole-gallery backups satisfy age, empty-gallery handling
and whether deferred startup work should retry later. Implementation may
prepare the default-off CPU contracts independently. It must not invent
these choices or enable automatic exports/retention to demonstrate progress.
