"""Managed sizing over scratch galleries; no activation, pruning or real secrets."""
from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from backend import backup_receipts, bundle_archive, bundle_capacity, bundles, grid
from backend.bundle_common import BundleError
from backend.schemas import GenerateRequest
from tests.bundle_helpers import _save, _snapshot


@pytest.fixture
def identity():
    return backup_receipts.ManagedIdentity(str(uuid4()), str(uuid4()), str(uuid4()))


@pytest.mark.parametrize("password", [None, "scratch-only password"])
def test_final_budget_uses_actual_post_snapshot_size_real_manifest_and_single_readback(
    store, monkeypatch, identity, password,
):
    _save()
    before = _snapshot(store)
    real_backup, real_capacity, real_verify = bundles._backup, bundle_capacity.managed_capacity, bundles._verify_archive
    checks, budgets, verifications = [], [], []

    def larger_snapshot(source, target):
        real_backup(source, target)
        with closing(sqlite3.connect(target)) as conn:
            with conn:
                conn.execute("CREATE TABLE capacity_fixture (data BLOB)")
                conn.execute("INSERT INTO capacity_fixture VALUES (zeroblob(1048576))")

    def budget(members, manifest_base, **kwargs):
        result = real_capacity(members, manifest_base, **kwargs)
        budgets.append((dict(members), json.loads(json.dumps(manifest_base)), kwargs, result))
        return result

    def verify(*args, **kwargs):
        verifications.append(True)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(bundles, "_backup", larger_snapshot)
    monkeypatch.setattr(bundle_capacity, "managed_capacity", budget)
    monkeypatch.setattr(bundles, "_check_space", lambda needs: checks.append(needs))
    monkeypatch.setattr(bundles, "_verify_archive", verify)
    result = bundles.export_bundle(keep_gallery=True, password=password, managed=identity)
    assert len(checks) == 2 and len(budgets) == len(verifications) == 1
    sizes, manifest_base, options, estimate = budgets[0]
    assert sizes["database.sqlite3"] >= 1048576
    assert manifest_base[backup_receipts.MARKER] == identity.marker() and manifest_base["files"] == {}
    assert options == {"manifest_name": "manifest.json", "encrypted": bool(password)}
    assert checks[-1] == [(bundles._directory(), estimate.archive_bytes, "write the managed bundle"),
                          (store.settings.db_path.parent, backup_receipts._MAX_BYTES, "record the managed bundle receipt")]
    assert result["bytes"] <= estimate.archive_bytes and result["managed"] is True
    with bundle_archive._open_archive(Path(result["path"]), password) as archive:
        assert archive.getinfo("database.sqlite3").file_size == sizes["database.sqlite3"]
        assert len(archive.read("manifest.json")) == estimate.manifest_bytes
    assert _snapshot(store) == before


@pytest.mark.parametrize("password", [None, "scratch-only password"])
@pytest.mark.parametrize("margin", [-1, 0])
def test_final_per_volume_boundary_preserves_old_archive_journal_and_gallery(
    store, monkeypatch, identity, password, margin,
):
    _save()
    old = bundles.export_bundle(keep_gallery=True, password=password, managed=identity)
    old_path = Path(old["path"])
    old_bytes = old_path.read_bytes()
    journal = backup_receipts.path_for(store.settings.db_path)
    previous_journal, gallery = journal.read_bytes(), _snapshot(store)
    checks = []
    real_check = bundles._check_space
    room = shutil.disk_usage(store.settings.db_path.parent)

    def check(needs):
        checks.append(needs)
        if len(checks) == 1:
            return  # Isolate the final, measured check from the early heuristic.
        assert len({path.stat().st_dev for path, _, _ in needs}) == 1
        with monkeypatch.context() as patch:
            patch.setattr(bundle_archive.shutil, "disk_usage", lambda where: room._replace(
                free=sum(size for _, size, _ in needs) + margin,
            ))
            real_check(needs)

    monkeypatch.setattr(bundles, "_check_space", check)
    next_identity = replace(identity, operation_id=str(uuid4()))
    if margin < 0:
        original_open = Path.open

        def no_archive_open(path, *args, **kwargs):
            if path.name.endswith((".zip.part", ".ldbundle.part")):
                pytest.fail("the final space refusal must precede opening an archive partial")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", no_archive_open)
        with pytest.raises(BundleError, match="not enough free space"):
            bundles.export_bundle(keep_gallery=True, password=password, managed=next_identity)
        assert journal.read_bytes() == previous_journal
        assert [item["name"] for item in bundles.list_bundles()] == [old_path.name]
    else:
        result = bundles.export_bundle(keep_gallery=True, password=password, managed=next_identity)
        assert result["managed"] is True and len(backup_receipts.load(store.settings.db_path, identity.gallery_id)) == 2
    assert len(checks) == 2 and _snapshot(store) == gallery and old_path.read_bytes() == old_bytes
    assert not list(bundles._directory().glob("*.part"))
    assert not list(store.settings.db_path.parent.glob(".bundle-*"))


def test_managed_compaction_early_budget_counts_snapshot_plus_two_extra_copies(store, monkeypatch, identity):
    _save()
    definition = {"job_id": "retained", "run_id": "retained", "created_at": 1.0, "origin": "grid",
                  "model_id": "test/model", "quantization": "fp8", "width": 256, "height": 256,
                  "batch": 1, "comparison": False, "metadata": {}}
    store.retain_run_job(definition, [grid.Cell(GenerateRequest(prompt="unfinished"), 1)])
    files = sum(path.stat().st_size for _, path in bundles._inventory())

    def first_check(needs):
        database_heuristic = needs[0][1] - files
        assert needs[1][1] == 3 * database_heuristic
        assert needs[2][1] == backup_receipts._MAX_BYTES
        raise BundleError("synthetic early refusal")

    monkeypatch.setattr(bundles, "_check_space", first_check)
    with pytest.raises(BundleError, match="synthetic early refusal"):
        bundles.export_bundle(keep_gallery=True, managed=identity)
    assert bundles.list_bundles() == []


def test_manual_export_keeps_its_original_single_check_and_does_not_use_managed_budget(store, monkeypatch):
    _save()
    checks = []
    monkeypatch.setattr(bundles, "_check_space", lambda needs: checks.append(needs))

    def forbidden(*args, **kwargs):
        pytest.fail("ordinary manual export must retain its existing path")

    monkeypatch.setattr(bundle_capacity, "managed_capacity", forbidden)
    result = bundles.export_bundle(keep_gallery=True)
    assert len(checks) == 1 and len(checks[0]) == 2 and "managed" not in result
    assert not backup_receipts.path_for(store.settings.db_path).exists()
