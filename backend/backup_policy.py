"""Pure automatic-backup decisions; no filesystem, app state or activation.

Copy facts are supplied by a verifier, not reconstructed from archive names.
These decisions authorize no deletion: a future coordinator must revalidate
ownership, identity, containment and leases under the gallery reservation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

Protection = Literal["unencrypted", "encrypted"]
CopyState = Literal["managed", "manual", "missing", "replaced", "unknown"]


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("backup clocks and verification times must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BackupPolicy:
    enabled: bool = False
    minimum_age: timedelta | None = None
    keep_last: int | None = None
    protection: Protection | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("backup enabled must be a boolean")
        if self.minimum_age is not None and (
            not isinstance(self.minimum_age, timedelta) or self.minimum_age <= timedelta(0)
        ):
            raise ValueError("backup minimum age must be a positive timedelta")
        if self.keep_last is not None and (type(self.keep_last) is not int or self.keep_last < 1):
            raise ValueError("backup retention must be a positive integer")
        if self.protection is not None and self.protection not in {"unencrypted", "encrypted"}:
            raise ValueError("backup protection must be explicit")
        if self.enabled and (self.minimum_age is None or self.keep_last is None or self.protection is None):
            raise ValueError("enabling backup needs an explicit age, retention and protection")


@dataclass(frozen=True)
class CopyFacts:
    """A caller-verified snapshot, not an archive receipt or journal format.

    `managed` means a published, verified, available, nonempty whole-gallery
    copy whose ownership and current archive identity both match its durable
    receipt. Other states confer no retention authority or age coverage.
    `archive_id` is an opaque identifier supplied by that verifier.
    """

    archive_id: str
    gallery_id: str | None
    destination_id: str | None
    verified_at: datetime | None
    protection: Protection | None
    state: CopyState
    download_leased: bool

    def __post_init__(self) -> None:
        if not isinstance(self.archive_id, str) or not self.archive_id.strip():
            raise ValueError("a backup copy needs an opaque archive identity")
        if self.state not in {"managed", "manual", "missing", "replaced", "unknown"}:
            raise ValueError("unknown backup copy state")
        if type(self.download_leased) is not bool:
            raise ValueError("a backup download lease must be explicit")
        if self.protection is not None and self.protection not in {"unencrypted", "encrypted"}:
            raise ValueError("unknown backup copy protection")
        for value in (self.gallery_id, self.destination_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("backup scope identities must be nonempty")
        if self.verified_at is not None:
            object.__setattr__(self, "verified_at", _utc(self.verified_at))
        if self.state == "managed" and any(value is None for value in (
            self.gallery_id, self.destination_id, self.verified_at, self.protection,
        )):
            raise ValueError("a managed backup needs verified scope, time and protection facts")


@dataclass(frozen=True)
class DueDecision:
    due: bool
    reason: Literal["disabled", "untrusted_inventory", "empty", "no_copy", "young", "due", "clock_backwards"]

    @property
    def clock_warning(self) -> bool:
        return self.reason == "clock_backwards"


@dataclass(frozen=True)
class RetentionPlan:
    candidates: tuple[str, ...]
    reason: Literal["disabled", "untrusted_inventory", "no_new_success", "clock_backwards", "planned"]


def _managed(
    copies: Sequence[CopyFacts], policy: BackupPolicy, gallery_id: str, destination_id: str,
) -> list[CopyFacts]:
    found = [copy for copy in copies if (
        copy.gallery_id == gallery_id
        and copy.destination_id == destination_id and copy.protection == policy.protection
    )]
    if len({copy.archive_id for copy in found}) != len(found):
        raise ValueError("duplicate archive identity in backup scope")
    return [copy for copy in found if copy.state == "managed"]


def _scope(gallery_id: str, destination_id: str, ownership_trusted: bool) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in (gallery_id, destination_id)):
        raise ValueError("backup planning needs explicit gallery and destination identities")
    if type(ownership_trusted) is not bool:
        raise ValueError("backup ownership trust must be a boolean")


def backup_due(
    policy: BackupPolicy, copies: Sequence[CopyFacts], *, now: datetime,
    gallery_id: str, destination_id: str, gallery_nonempty: bool, ownership_trusted: bool,
) -> DueDecision:
    """Check supplied current automatic copies only; never advance success age.

    False trust covers an absent/damaged journal or incomplete ownership
    inventory. Empty/skipped/failed attempts are never successful copies.
    A coordinator must recheck these facts after acquiring its reservation.
    """
    now = _utc(now)
    _scope(gallery_id, destination_id, ownership_trusted)
    if type(gallery_nonempty) is not bool:
        raise ValueError("backup gallery content must be an explicit boolean fact")
    if not policy.enabled:
        return DueDecision(False, "disabled")
    if not ownership_trusted:
        return DueDecision(False, "untrusted_inventory")
    if not gallery_nonempty:
        return DueDecision(False, "empty")
    copies = _managed(copies, policy, gallery_id, destination_id)
    if not copies:
        return DueDecision(True, "no_copy")
    latest = max(copy.verified_at for copy in copies if copy.verified_at is not None)
    elapsed = now - _utc(latest)
    if elapsed < timedelta(0):
        return DueDecision(False, "clock_backwards")
    assert policy.minimum_age is not None
    return DueDecision(True, "due") if elapsed >= policy.minimum_age else DueDecision(False, "young")


def retention_plan(
    policy: BackupPolicy, copies: Sequence[CopyFacts], *, now: datetime,
    gallery_id: str, destination_id: str, new_success: str | None, ownership_trusted: bool,
) -> RetentionPlan:
    """Plan identifiers only, after a new copy and durable receipt exist.

    A failed, empty or skipped attempt supplies no `new_success`. Keep the
    newest N confirmed copies, the newly published one even across a clock
    correction, and every leased copy. Other scopes/protection sets stay
    untouched; no earlier protection mode is implicitly pruned.
    """
    now = _utc(now)
    _scope(gallery_id, destination_id, ownership_trusted)
    if not policy.enabled:
        return RetentionPlan((), "disabled")
    if not ownership_trusted:
        return RetentionPlan((), "untrusted_inventory")
    current = _managed(copies, policy, gallery_id, destination_id)
    if new_success is None or not any(copy.archive_id == new_success for copy in current):
        return RetentionPlan((), "no_new_success")
    if any(copy.verified_at is not None and _utc(copy.verified_at) > now for copy in current):
        return RetentionPlan((), "clock_backwards")
    newest = sorted(current, key=lambda copy: (copy.verified_at, copy.archive_id), reverse=True)
    assert policy.keep_last is not None
    keep = {copy.archive_id for copy in newest[:policy.keep_last]} | {new_success}
    return RetentionPlan(tuple(
        copy.archive_id for copy in reversed(newest) if copy.archive_id not in keep and not copy.download_leased
    ), "planned")
