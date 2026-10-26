# Plan: automatic whole-gallery backups

**Status, 2026-10-26: explicit default-off plaintext startup backups,
configuration/status UI and owned preload/shutdown sequencing are implemented.
Managed receipts, capacity budgets and opt-in shared-lock helpers are built;
`GAL#6` remains open for shared-lock installation, retention and an OS-backed
protection option.** This expands
[R8](review-2026-10-25.md#r8-automatic-whole-gallery-backup). Tests export and
restore scratch galleries only; no real gallery or credential store was touched.
Automatic backups default off. No real gallery was activated. The implemented
plaintext option needs explicit consent, age and count at its displayed
destination; no values or protection mode are selected for Matt. Automatic
cleanup is absent, so the saved retention count is explicitly not enforced.

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

## Explicit startup/configuration slice

`backup_settings.py` adds a strict v2 activation envelope at the existing
database-specific policy path. Missing settings create nothing; every v1 draft,
even `enabled: true`, stays inactive and requires new confirmation. Reads are
bounded to 64 KiB and writes fit the same limit. Explicit saves use the gallery
reservation, revision comparison and an atomic same-directory temporary. A
failed or uncertain save rereads the policy before another attempt. No raw
payload or exception is returned by the configuration route.

Enabling requires fresh acknowledgement of unencrypted images/prompts, positive
whole-hour age and count, and the exact displayed destination. Only that save
creates stable gallery/destination UUIDs; a destination change rotates its managed
set without adopting old archives. An existing journal for another or damaged
namespace refuses. Disabling remains possible when the destination is unavailable.
Unknown credentials/protection fields are rejected with a fixed validation error;
no OS provider or plaintext fallback is installed.

`BackupService` reads settings off-loop after normal startup settlement, reserves
the existing bundle manager synchronously before serving changing requests and
invokes one due/export attempt. Its tracked supervisor waits for that attempt
before optional preload; shutdown cancels pending preload and drains admitted
file-writing work through the existing manager. Disabled/missing settings create
no IDs, directories, companions or archives. Startup still makes no retry loop.

`GET /api/gallery/automatic-backup` is a cached, model-free status even during
maintenance. An explicit POST to its `/reload` reads bounded settings off-loop,
without journal hashes or an export, and retains its read guard through request
cancellation. PUT configures the next server start; Save itself creates no backup.
The collapsed Bundles section keeps drafts/revisions separate from polls, handles
external changes and late replies, and leaves manual password/count fields alone.
Automatic outcomes use their own response shape; verified publication is required
before success wording. Age counts only currently matching managed copies,
skips an empty gallery and warns about backward clocks. UI copy states that all
existing copies are kept and the requested N is not yet enforced.

The new activation/settings/API batch has **43 tests**; the combined
manager/admission/docs/classic-scope batch passed **137 tests**. Seven affected
production modules passed the pinned type check, and independent source review
accepted the final fixes. Scratch acceptance covers two server starts, verified
restore of two checkpoints, stars, bin, refine lineage and mask data, preservation
of a manual archive, preload ordering and shutdown while a writer is admitted.
These are synthetic storage/lifecycle checks, not real gallery, OS credential,
device or deletion acceptance.

Before the final receipt/configuration overlap fix, the complete offline CPU gate passed **4,896 tests, eight skips and 75 existing
warnings** in **223.93 seconds**, with two workers. The pinned whole-source type
gate checked **120 files with zero errors/warnings**, and repository Ruff passed.
The final fix snapshots cache state under a short lock, reads receipts off-loop
against a captured attempt and publishes its timestamp only if that same attempt
and scope still own the service. The affected integration group then passed all
**106 tests**; the final barrier at the actual receipt-read boundary passed its
isolated check too. Narrow types and Ruff stayed clean. The regression disables
settings after worker admission releases but before the receipt read finishes,
and proves preload still completes without an old timestamp or supervisor error.
The automatic-settings browser smoke passed on an isolated Edge fake, including
1400/600 px containment and stationary Save under a synthetic long warning.
The four retained screenshots were inspected. Browser configuration itself
creates no archive and performs no server restart; scratch lifecycle/restore
acceptance is covered by the CPU integration cases above.

The existing full manual Bundles smoke passed all **133 original assertions** in
32 browser seconds. Its first full run failed two expectations at a duplicate
saved-import step; a bounded diagnostic and then the fresh full run did not
reproduce them. No product code or assertion was loosened; the initial cause is
unproven. Reads of that first run's result/fake-log artifacts were denied and
stopped; its separate stdout receipt supplied diagnostics. No further read in
that artifact tree or permission workaround was attempted.

After the subsequent loader-ownership integration, the final complete offline
gate passed **4,920 tests with eight skips** in **225.12 seconds**, covering the
receipt/configuration overlap fix too. All **120** production/tool sources pass
the pinned type gate, and repository Ruff passes. No additional backup behavior
was activated by that validation.

## Activation, age and saved state

Use gallery-scoped, atomically saved backup preferences, independent of
the four model/startup choices. An explicit checkpoint override must not
disable the gallery's protection or accidentally enable it. A dedicated
versioned sidecar beside the database is suitable; reuse the existing
atomic-write discipline without putting unrelated fields in
`StartupPreferences`. The implemented v1 sidecar is
`local-diffusion.backup-policy-<database filename>.json`, isolating databases
even when they share a parent. Its complete schema is `version`, `enabled`,
nullable `minimum_age_us`, `keep_last` and `protection`. Integer microseconds
preserve every timedelta exactly without floating-point age rounding.
Retention has a shared engineering ceiling of 10,000 copies, with no count
selected by default. An enabled policy requires explicit positive age,
retention and protection. Missing settings read disabled; malformed,
duplicate/unknown-field, nonfinite, oversized or newer data returns a fixed
repairable warning and disabled policy, without changing its file. Saves
validate first, fsync an owned temporary and atomically replace this
database's sidecar. This v1 draft contract has no credential field and does not
activate; the lifecycle uses only the explicit v2 envelope described above.

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
snapshot/WAL and any retained-job snapshot compaction. Managed exports now
also budget ZIP member overhead, encryption framing and an atomic receipt
temporary, using the actual compacted snapshot before archive creation.
These are logical-byte preflights, not an allocation or disk-fit guarantee;
filesystem overhead, SQLite growth/temp placement and other writers remain
limits. Ordinary manual exports retain their existing estimate.
A full or unavailable destination refuses before
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
Pyright check reports zero errors/warnings. The subsequent preference slice
and shared retention-ceiling tests passed 112 pure scratch checks covering
strict bounded reads, database isolation, exact durations and injected
write/fsync/replace failures that preserve previous bytes. Ruff and pinned
Pyright pass. No real gallery or archive was touched by these
policy/preference checks.

**Managed-export evidence slice.** The internal optional `managed` argument
accepts canonical nonzero gallery/destination/operation UUIDs only for a
whole-gallery export that keeps its source. A reserved `managed_backup`
manifest marker is checked against that identity during the existing one
verifying read. A forward-only outer writer computes SHA-256 and byte count
over the actual ZIP or ciphertext, with no second archive read. Ordinary
exports retain their writer/manifest and create no receipt journal.

After closing the writer, its partial's identity must still match before
verification. Publication claims a fresh name without overwrite; success
and failure cleanup remove only a partial whose device/inode still belongs
to that worker. Under the existing gallery reservation, a database-specific
bounded atomic journal then records scope, exact basename, UTC verification,
protection, raw digest/count and file identity. Failed journal publication
keeps prior bytes and the new archive, reports a fixed warning and leaves
that archive unmanaged. Malformed/missing ownership is never reconstructed
from a name or marker. Current-file checks require resolved destination,
regular-file identity and a full raw digest, including for encrypted files
without obtaining their password. They return a snapshot, never deletion
authority. Portable basenames reject Windows device names.

Focused validation passed 139 bundle/receipt/backup/password/merge/staging
checks before review hardening, then all 62 managed/receipt cases after the
final fixes. Ruff and narrow pinned Pyright pass. Independent review caught
pre-verification partial replacement, successful cleanup of a recreated
unowned partial and reserved Windows names; regressions cover each. Plain
and encrypted scratch copies restore through the ordinary importer with
bin, stars and lineage intact, and verify exactly once. Windows path/handle
timestamp differences are handled without substituting metadata for the
digest. Linux publication and actual OS credential delivery remain untested.

There are no automatic-backup startup/API/UI callers, credential lookup
or retention deletion. Shared-lock installation, activation/recovery decisions
and end-to-end startup acceptance below remain
unimplemented. The inactive coordinator described below derives age from
currently matching receipts rather than maintaining a separate success clock.

**Per-process download lease slice.** Saved bundle downloads now use
`bundle_leases.LeasedFileResponse`. It acquires a resolved-path reader
lease when the ASGI response starts, then releases in finally across normal
body, ranges, HEAD, errors, cancellation and background completion. It
disables pathsend handoff rather than release before the server finishes
transferring a path. A mutex-protected registry supplies reader/exclusive
exclusion and idempotent cross-thread release; ordinary bundle maintenance
still permits downloads. All 13 pure focused tests passed, and the combined
lease/existing bundle HTTP suite passed 41 tests, with Ruff and narrow
Pyright clean. Independent review found no actionable gaps.

Default downloads cover cooperating threads in this process. Shared-lock
installation across all cooperating processes remains mandatory before
retention deletion can be enabled. The exclusive claim primitive has no
mutation caller. Startup, activation and retention stages remain unconnected.

**Opt-in shared-lock slice.** An explicitly constructed
`LeaseRegistry(cross_process=True)` adds native nonblocking shared/exclusive
locks on persistent `<archive basename>.lease.lock` companions. Windows
uses `LockFileEx`; the CRT's similarly named read lock is exclusive
([Microsoft's locking reference](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking?view=msvc-170)).
The companion is never truncated, initialized, replaced or deleted. Opened
handle, current path and parent identities are checked; linked, reparse,
nonregular and hardlinked companions refuse. Windows handles deny delete
sharing while held. Only lock contention becomes a busy response; access,
identity and cleanup failures remain errors without an unlocked fallback.

Registries share process-wide reader/exclusive exclusion and reader hints.
An acquired native shared handle remains held until every local reader,
including a default-mode reader, releases. Cross-thread release is
idempotent; uncertain close retains local exclusion without an implicit
retry. Original body failures retain release diagnostics. Defaults retain
non-writing downloads, so a readable recovery destination need not become
writable for this inactive feature. A future shared-mode installer must run
while quiescent and before admitting streams/retention everywhere.

All 51 focused scratch checks pass, including actual Windows shared readers,
exclusive contention, process exit and mixed-mode lifetime handshakes through
owned subprocesses. No native access denial occurred. Injected open/lock/
unlock/close and identity failures exercise fail-closed paths. Ruff and
narrow pinned Pyright pass; independent review accepted ownership and the
final process-wide reader hints. Linux/fork, network filesystems and mixed
old/default/noncooperating processes remain unqualified. No shared installer
or deletion caller exists; a lease never substitutes for receipt, destination
and archive identity checks at a future mutation boundary.
The combined lease/coordinator/bundle HTTP integration check passed all 114
tests; this ordinary fixture run includes its existing CUDA-availability guard,
with no model loads, tensor execution or generation.

**Managed capacity slice.** `bundle_capacity.managed_capacity` uses current
member stat sizes, the actual post-compaction SQLite snapshot and a manifest
skeleton with real names/sizes/ownership marker and fixed-length SHA-256
placeholders. Its ZIP_STORED bound includes local/central headers, UTF-8
filename lengths, maximal ZIP64 extras, forward-only descriptors and end
records. Encrypted output adds the current header and exact chunk framing;
it does not reserve or write a second plaintext archive. The receipt budget
reserves the journal's validated 16 MiB cap for its atomic temporary beside
the database. Existing volume grouping sums shared-volume needs.

Managed export checks these final budgets before opening its archive
partial, with no extra content read, verification pass or key derivation.
Its early DB+WAL check remains a heuristic; when retained cells require
VACUUM it allows three snapshot-size equivalents, since SQLite documents
up to two additional copies while the snapshot already exists
([SQLite VACUUM](https://www.sqlite.org/lang_vacuum.html)). Actual filesystem
allocation, SQLite growth and external changes still prevent a fit promise.
The ordinary manual export check and writer remain unchanged.

All 110 focused math/managed-export/backup/password/receipt checks passed,
including 35 new cases. They compare actual plain/encrypted output against
bounds, force ZIP64 branches on small buffers, cover chunk boundaries and
virtual large integer sizes, and inject shortages before archive creation.
Previous archives, journal and live scratch gallery survive refusal; successful
copies still verify once. Two initial enlarged-snapshot fixtures left their
SQLite connection open and hit Windows sharing refusals at temporary cleanup;
their published archive/receipt outcomes were checked before fixing the
fixture with explicit close and rerunning in fresh scratch. The original
cleanup targets were not retried. Ruff and narrow pinned Pyright pass;
no real gallery, password store or GPU work was used.

**Inactive one-attempt coordinator.** `backup_coordinator.BackupCoordinator`
accepts an explicit policy, gallery/destination/operation identities, aware
clock, transient password and ownership-trust fact. Disabled policies do
no I/O or admission. Enabled calls reserve the existing `BundleManager`
synchronously and run the due check and ordinary verified whole-gallery
export in one worker under the existing cross-process gallery reservation.
Untrusted ownership, mismatched sources/destinations, unusable journals and
protection failures produce fixed outcomes without plaintext fallback.
Both indexed records and unindexed gallery images count as nonempty.

Recorded operation IDs skip before hashing or export, including retries
through a new coordinator. Age uses only currently matching nonempty
managed receipts for this destination and protection. A published archive
without a saved receipt reports unmanaged; no second read-back or independent
success timestamp is added. Refused calls preserve the latest admitted
outcome, including the interval between worker completion and admission
release. Lease tokens reject repeat entry and entry after release.

Shutdown drains the admitted worker. Cancellation before its supervisor's
first turn releases admission, reports a fixed terminal outcome and cleans
its owned upload once; caller cancellation still propagates. Tests and
independent review cover these lifetime boundaries. Executor submission can
queue a callable before raising; a short submission guard permits it only
after the supervisor receives its Future, preventing an abandoned writer.
Submission failure releases admission and records a fixed terminal reason.
All 77 focused coordinator/lease/bundle HTTP checks pass, with Ruff and narrow
pinned Pyright clean. No startup/preload,
credential provider, recurring retry, UI or pruning is connected.

1. **Policy and journal contracts, CPU only.** Implement validated default-off
   preferences and a pure due/retention
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
3. **Protection and UI.** Require explicit choices before activation; never
   select a mode or values for Matt. For the OS option, unit tests use a fake credential provider and integration
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

This is an activation boundary, not a block on writing configurable software.
Age, count and destination can remain required inputs with no selected values;
plaintext needs informed explicit consent and encrypted activation needs an
actual supported credential provider. Existing conservative empty-gallery,
manual-age and no-retry semantics are versioned first-slice behavior, not
inferred personal preferences. Source work and synthetic lifecycle checks may
continue without activating any real gallery or credential store.
