"""Pure backup age and conservative managed-retention contracts."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone, tzinfo

import pytest

from backend.backup_policy import MAX_KEEP_LAST, BackupPolicy, CopyFacts, backup_due, retention_plan

NOW = datetime(2026, 10, 26, 12, tzinfo=timezone.utc)
SCOPE = {"gallery_id": "gallery-a", "destination_id": "destination-a", "ownership_trusted": True}
POLICY = BackupPolicy(enabled=True, minimum_age=timedelta(hours=12), keep_last=2, protection="encrypted")


def copy(name: str, hours_old: float, **changes) -> CopyFacts:
    return replace(CopyFacts(
        archive_id=name, gallery_id="gallery-a", destination_id="destination-a",
        verified_at=NOW - timedelta(hours=hours_old), protection="encrypted", state="managed", download_leased=False,
    ), **changes)


def due(copies=(), *, policy=POLICY, now=NOW, **changes):
    return backup_due(policy, copies, now=now, gallery_nonempty=True, **(SCOPE | changes))


def retained(copies, new_success, *, policy=POLICY, now=NOW, **changes):
    return retention_plan(policy, copies, now=now, new_success=new_success, **(SCOPE | changes))


def test_default_policy_does_no_work_and_requires_explicit_choices_to_enable():
    assert due(policy=BackupPolicy()).reason == "disabled"
    assert retained([copy("a", 1)], "a", policy=BackupPolicy()).candidates == ()
    for missing in ("minimum_age", "keep_last", "protection"):
        with pytest.raises(ValueError, match="explicit age, retention and protection"):
            replace(POLICY, **{missing: None})


@pytest.mark.parametrize("changes", [
    {"minimum_age": timedelta(0)}, {"minimum_age": timedelta(seconds=-1)},
    {"keep_last": 0}, {"keep_last": -1}, {"keep_last": True},
    {"protection": "plaintext-fallback"}, {"enabled": 1},
])
def test_invalid_policy_never_becomes_a_working_plan(changes):
    with pytest.raises(ValueError):
        replace(POLICY, **changes)


@pytest.mark.parametrize("keep_last", [1, MAX_KEEP_LAST - 1, MAX_KEEP_LAST])
@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_retention_accepts_both_bounds_and_the_value_below_the_ceiling(keep_last, enabled):
    assert replace(POLICY, enabled=enabled, keep_last=keep_last).keep_last == keep_last


@pytest.mark.parametrize("keep_last", [MAX_KEEP_LAST + 1, 10**100])
@pytest.mark.parametrize("enabled", [False, True])
def test_retention_ceiling_also_applies_to_disabled_policies(keep_last, enabled):
    with pytest.raises(ValueError, match=f"from 1 to {MAX_KEEP_LAST}"):
        replace(POLICY, enabled=enabled, keep_last=keep_last)


def test_due_uses_available_verified_success_and_exact_utc_boundary():
    latest = copy("latest", 12)
    assert due([latest], now=NOW - timedelta(microseconds=1)).reason == "young"
    assert due([latest]).due
    assert due([latest], now=NOW + timedelta(microseconds=1)).due
    offset = timezone(timedelta(hours=5, minutes=30))
    assert due([replace(latest, verified_at=latest.verified_at.astimezone(offset))], now=NOW.astimezone(offset)).due


def test_repeated_wall_hour_uses_utc_order_for_due_and_newest_retained_copy():
    class RepeatedHour(tzinfo):
        # One overlapping wall hour, without an OS timezone database.
        def utcoffset(self, value):
            return timedelta(hours=-5 if value is not None and value.fold else -4)

        def dst(self, value):
            return timedelta(0)

    zone = RepeatedHour()
    older = copy("older", 0, verified_at=datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=0))
    later = copy("later", 0, verified_at=datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=1))
    new = copy("new", 0, verified_at=datetime(2026, 11, 1, 7, tzinfo=timezone.utc))
    assert older.verified_at == datetime(2026, 11, 1, 5, 45, tzinfo=timezone.utc)
    assert later.verified_at == datetime(2026, 11, 1, 6, 15, tzinfo=timezone.utc)
    assert older.verified_at.tzinfo is later.verified_at.tzinfo is timezone.utc
    policy = replace(POLICY, minimum_age=timedelta(hours=1))
    assert due([older, later], policy=policy, now=datetime(2026, 11, 1, 6, 45, tzinfo=timezone.utc)).reason == "young"
    assert due([older, later], policy=policy, now=datetime(2026, 11, 1, 7, 15, tzinfo=timezone.utc)).due
    assert retained([older, later, new], "new", now=new.verified_at).candidates == ("older",)


def test_backward_clock_warns_and_does_no_catch_up_work():
    decision = due([copy("future", -1)])
    assert not decision.due and decision.clock_warning
    assert retained([copy("old", 50), copy("new", -1)], "new").reason == "clock_backwards"


def test_empty_gallery_and_untrusted_journal_never_advance_or_prune():
    copies = [copy("old", 80), copy("new", 0)]
    decision = backup_due(POLICY, copies, now=NOW, gallery_nonempty=False, **SCOPE)
    assert not decision.due and decision.reason == "empty"
    assert not due(copies, ownership_trusted=False).due
    assert retained(copies, "new", ownership_trusted=False).candidates == ()
    assert copies == [copy("old", 80), copy("new", 0)]


def test_truthy_strings_cannot_replace_validated_trust_or_gallery_facts():
    with pytest.raises(ValueError, match="boolean"):
        due(ownership_trusted="yes")
    with pytest.raises(ValueError, match="boolean"):
        retained([copy("new", 0)], "new", ownership_trusted=1)
    with pytest.raises(ValueError, match="boolean"):
        backup_due(POLICY, (), now=NOW, gallery_nonempty="false", **SCOPE)
    with pytest.raises(ValueError, match="identities"):
        due(gallery_id=" ")


@pytest.mark.parametrize("state", ["manual", "missing", "replaced", "unknown"])
def test_unconfirmed_or_manual_copy_never_covers_age_or_becomes_a_candidate(state):
    other = copy("auradream-automatic-looking.zip", 100, state=state)
    assert due([other]).reason == "no_copy"
    assert retained([other, copy("one", 10), copy("new", 0)], "new").candidates == ()
    assert retained([copy("old", 100), other], other.archive_id).reason == "no_new_success"


@pytest.mark.parametrize("changes", [
    {"gallery_id": "other-gallery"}, {"destination_id": "old-destination"}, {"protection": "unencrypted"},
])
def test_other_managed_sets_never_count_toward_age_or_retention(changes):
    other = copy("new", 100, **changes)
    assert due([other]).reason == "no_copy"
    assert retained([other, copy("new", 0)], "new", policy=replace(POLICY, keep_last=1)).candidates == ()


def test_failed_empty_or_skipped_attempt_has_no_new_verified_success():
    copies = [copy("last-good", 40), copy("older-good", 80)]
    assert retained(copies, None, policy=replace(POLICY, keep_last=1)).reason == "no_new_success"
    assert retained(copies, "partial-or-unreceipted", policy=replace(POLICY, keep_last=1)).candidates == ()
    assert due(copies).due


def test_retention_keeps_newest_n_and_protects_leases_without_predeleting():
    copies = [copy("oldest", 80), copy("leased", 60, download_leased=True), copy("second", 20), copy("new", 0)]
    assert retained(copies, "new").candidates == ("oldest",)
    assert retained(copies, None).candidates == ()
    assert retained(copies, "new", policy=replace(POLICY, keep_last=4)).candidates == ()
    assert copies[0].archive_id == "oldest"  # Planning never mutates the inventory.


def test_new_copy_is_kept_even_when_its_clock_is_older_than_existing_copies():
    copies = [copy("new-publication", 60), copy("old-publication", 20)]
    assert retained(copies, "new-publication", policy=replace(POLICY, keep_last=1)).candidates == ()


def test_equal_verification_times_have_stable_order_and_always_keep_new_copy():
    copies = [copy("a", 20), copy("b", 20), copy("c", 20)]
    expected = retained(copies, "a", policy=replace(POLICY, keep_last=1))
    assert expected.candidates == ("b",)
    assert retained(list(reversed(copies)), "a", policy=replace(POLICY, keep_last=1)) == expected


def test_ambiguous_managed_identity_and_naive_clocks_are_rejected():
    with pytest.raises(ValueError, match="duplicate archive"):
        retained([copy("same", 20), copy("same", 0)], "same")
    with pytest.raises(ValueError, match="timezone-aware"):
        due(now=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="timezone-aware"):
        copy("bad", 0, verified_at=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="verified scope"):
        copy("unbound", 0, gallery_id=None)


@pytest.mark.parametrize("state", ["replaced", "unknown"])
def test_conflicting_same_scope_identity_is_rejected_before_state_filtering(state):
    copies = [copy("same", 30), copy("same", 0, state=state)]
    with pytest.raises(ValueError, match="duplicate archive"):
        due(copies)
    with pytest.raises(ValueError, match="duplicate archive"):
        retained(copies, "same", policy=replace(POLICY, keep_last=1))
