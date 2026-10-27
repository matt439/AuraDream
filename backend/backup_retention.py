"""Explicit-consent post-success cleanup; never a startup/retry loop of its own."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import backup_receipts as receipts, bundle_delete, bundle_leases, bundle_mode, gallery_lock
from .backup_policy import BackupPolicy, CopyFacts, retention_plan
from .backup_settings import RetentionPermission


@dataclass(frozen=True)
class RetentionOutcome:
    state: str = "disabled"
    reason: str = "not_enabled"
    marked: int = 0
    retained: int = 0  # Older candidates not marked, excluding the protected N.
    pending: bool = False
    warning: str | None = None
    uncertain: bool = False


_WARNING = "The new backup succeeded, but cleanup did not finish. Earlier removal may be pending or uncertain; inspect the journal before enabling another cleanup attempt."


def _scope(receipt: receipts.Receipt, identity: receipts.ManagedIdentity, permission: RetentionPermission) -> bool:
    return (receipt.identity.gallery_id == identity.gallery_id and receipt.identity.destination_id == identity.destination_id
            and receipt.directory == permission.directory and receipt.protection == permission.protection
            and bool(receipt.records or receipt.images))


def prune_after_success(
    db_path: Path, permission: RetentionPermission | None, policy: BackupPolicy,
    identity: receipts.ManagedIdentity, new_success: receipts.Receipt, *, now: datetime,
    previous_facts: list[CopyFacts],
) -> RetentionOutcome:
    """Called after exact durable publication under the existing reservation.

    Protected older copies are requalified while shared claims remain held.
    The just-verified new archive is not read back again. Candidate byte proof
    happens once on its bound mutation handle, after durable pending intent.
    """
    if permission is None:
        return RetentionOutcome()
    marked, retained = 0, 0
    try:
        with gallery_lock.hold(db_path):
            if (permission.gallery_id != identity.gallery_id or permission.destination_id != identity.destination_id
                    or permission.keep_last != policy.keep_last or permission.protection != policy.protection
                    or permission.directory != str(permission.mode.directory)
                    or bundle_mode.discover(permission.mode.directory) != permission.mode):
                return RetentionOutcome("blocked", "permission_changed", warning="The saved deletion permission or installed shared mode changed. Cleanup is inactive until it is confirmed again.")
            state = receipts.load_state(db_path, identity.gallery_id)
            if state.pending is not None:
                return RetentionOutcome("blocked", "pending_intent", pending=True, warning=_WARNING)
            if (new_success not in state.receipts or new_success.identity != identity
                    or not _scope(new_success, identity, permission)):
                return RetentionOutcome("blocked", "new_receipt_unconfirmed", warning=_WARNING)
            scoped = [item for item in state.receipts if _scope(item, identity, permission)]
            if any(item.verified_at > now for item in scoped):
                return RetentionOutcome("blocked", "clock_backwards", warning="The clock moved backward. The backup succeeded and cleanup was skipped.")
            by_id = {item.identity.operation_id: item for item in scoped}
            prior = {item.archive_id: item for item in previous_facts}
            facts: dict[str, CopyFacts] = {}
            # The new copy's durable exporter proof supplies its byte identity;
            # its shared claim prevents a cooperating mutation through cleanup.
            with ExitStack() as protected:
                protected.enter_context(bundle_leases.reading(permission.mode.directory / new_success.name))
                if receipts.FileStamp.of((permission.mode.directory / new_success.name).lstat()) != new_success.stamp:
                    return RetentionOutcome("blocked", "new_receipt_unconfirmed", warning=_WARNING)
                facts[identity.operation_id] = CopyFacts(identity.operation_id, identity.gallery_id, identity.destination_id,
                                                       new_success.verified_at, new_success.protection, "managed", True)
                protected_ids = {identity.operation_id}
                confirmed = 0
                invalid: set[str] = set()
                # Keep the newest N CURRENT matching copies, plus the new copy
                # even across timestamp ties. Busy/unknown copies never fill N.
                for item in sorted(scoped, key=lambda value: (value.verified_at, value.identity.operation_id), reverse=True):
                    if confirmed >= permission.keep_last:
                        break
                    operation = item.identity.operation_id
                    if operation == identity.operation_id:
                        confirmed += 1
                        continue
                    try:
                        with ExitStack() as candidate:
                            owner = candidate.enter_context(bundle_leases.reading(permission.mode.directory / item.name))
                            if not receipts._matches_current(item, permission.mode.directory, owner):
                                invalid.add(operation)
                                continue
                            protected.enter_context(candidate.pop_all())
                    except bundle_leases.LeaseBusy:
                        invalid.add(operation)
                        continue
                    facts[operation] = CopyFacts(operation, identity.gallery_id, identity.destination_id,
                                                item.verified_at, item.protection, "managed", True)
                    protected_ids.add(operation)
                    confirmed += 1
                for operation in by_id:
                    if operation in facts or operation in invalid:
                        continue
                    previous = prior.get(operation)
                    if previous is not None and previous.state == "managed":
                        facts[operation] = previous
                plan = retention_plan(policy, list(facts.values()), now=now, gallery_id=identity.gallery_id,
                                      destination_id=identity.destination_id, new_success=identity.operation_id,
                                      ownership_trusted=True)
                # The pure planner excludes already-leased copies entirely.
                # Count their excess separately without making them candidates.
                known_excess = set(facts) - protected_ids
                retained = len(known_excess - set(plan.candidates))
                for index, operation in enumerate(plan.candidates):
                    item = by_id[operation]
                    bound = bundle_delete.bind_managed_archive(item, permission.mode, item.identity)
                    intent = receipts.PendingDeletion(item, str(uuid4()), identity.operation_id, permission.mode)
                    begun = False
                    counted = False
                    try:
                        with bound:
                            receipts.begin_deletion(db_path, intent)
                            begun = True
                            bound.mark_for_deletion()
                        marked += 1
                        counted = True
                        # Normal context exit, not just archive-close success,
                        # proves all pins/claim closed before journal retirement.
                        receipts.complete_deletion(db_path, intent)
                    except bundle_leases.LeaseBusy:
                        if begun:
                            raise  # A post-intent failure is never a harmless skip.
                        retained += 1
                        continue
                    except Exception:
                        mark = bound.outcome.mark
                        if begun and mark in ("not_requested", "refused") and bound.closed_normally:
                            receipts.restore_deletion(db_path, intent, mark=mark)
                        elif mark == "accepted" and not counted:
                            marked += 1
                        remaining = len(plan.candidates) - index - 1
                        not_marked = int(mark in {"not_requested", "refused"})
                        current = receipts.load_state(db_path, identity.gallery_id)
                        return RetentionOutcome("warning", "cleanup_incomplete", marked, retained + remaining + not_marked,
                                                current.pending is not None, _WARNING)
                return RetentionOutcome("warning" if retained else "succeeded", "busy_copies" if retained else "completed",
                                        marked, retained, warning="Some older copies are in use and were kept. The count can exceed the requested N." if retained else None)
    except Exception:
        # Do not downgrade the already verified backup or expose native errors.
        try:
            pending = receipts.load_state(db_path, identity.gallery_id).pending is not None
            reason, uncertain = "cleanup_incomplete", False
        except Exception:
            pending, reason, uncertain = False, "journal_unusable", True
        return RetentionOutcome("warning", reason, marked, retained, pending, _WARNING, uncertain)
