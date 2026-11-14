"""The checkpoint delete's converted-tree removal (`GEN#72` stage 2), backend
only, on scratch caches.

docs/maintainer/plan-model-delete.md, "How files are removed > Converted
tree", is the spec: under the loaders' own lock, the tree is renamed into
exactly `single_file._is_work`'s shape and removed from there with every
stale work directory, the `.lock` staying. Every injected failure
(`os.rename`, `shutil.rmtree` raising Windows' sharing violation) must
leave nothing but `_is_work` directories, and a later load converts cleanly
in the scratch cache, in `tests/test_converter_cache.py`'s style. No route
reaches the delete yet; stage 3 adds it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

from backend import keep_awake, model_cache_delete as delete, model_cache_storage, model_info
from backend.backends import anima_loading, netayume_loading, newbie_loading, single_file
from backend.encoder_table import TokenizerSource
from backend.refusal_codes import (
    MODEL_CACHE_CHANGED,
    MODEL_DELETE_BUSY,
    MODEL_DELETE_HELD,
    MODEL_DOWNLOADING,
    MODEL_FILES_IN_USE,
    MODEL_SELECTED,
)
from tests.model_cache_fixtures import link_folder, scratch_caches, tree_state

WAI = single_file.WAI_ID
TURBO = anima_loading.MODEL_ID
AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
NETAYUME, NEWBIE = netayume_loading.MODEL_ID, newbie_loading.MODEL_ID
SHARING_VIOLATION = 32  # ERROR_SHARING_VIOLATION: another process has the file open
_real_rmtree = shutil.rmtree


def lock_left_alone(lock: Path) -> None:
    """The delete never removes a `.lock`. filelock itself removes one on
    release on Windows (`WindowsFileLock`), never on Linux, and the loaders
    recreate it either way."""
    if sys.platform != "win32":
        assert lock.is_file()
    assert not lock.exists() or lock.stat().st_size == 0


@pytest.fixture
def caches(tmp_path, monkeypatch):
    # Converted trees are plain folders on every platform: one layout.
    return scratch_caches(tmp_path, monkeypatch, "copy")


def in_use(path) -> PermissionError:
    """What Windows raises for a file another process holds open."""
    return PermissionError(13, "The process cannot access the file because it is being used by another process",
                           str(path), SHARING_VIOLATION)


def run(model: str, **context) -> delete.Outcome:
    """Plan, then delete with that plan's pin, as stage 3's route will."""
    planned = delete.plan(model, delete.PlanContext(**context))
    return delete.delete(model, planned.pin, delete.PlanContext(**context))


def converted_state(caches) -> dict[str, tuple]:
    """The converted root byte for byte, but for the zero-byte `.lock` files
    a lock leaves (they stay by design)."""
    return {key: value for key, value in tree_state(caches.converted).items() if not key.endswith(".lock")}


def folders(caches) -> set[str]:
    return {path.name for path in caches.converted.iterdir() if path.is_dir()}


def only_work_left(caches, target: Path, kept: set[str]) -> list[Path]:
    """Every folder in the converted root beside `kept` is one of `target`'s
    work directories; never a marker-less target."""
    extra = [path for path in caches.converted.iterdir() if path.is_dir() and path.name not in kept]
    assert all(single_file._is_work(path, target) for path in extra), extra
    assert not os.path.lexists(target)
    return extra


def stale_work(target: Path, suffix: str = "abcd1234") -> Path:
    work = target.parent / f"{target.name}-{suffix}"
    (work / "tree" / "unet").mkdir(parents=True)
    (work / "tree" / "unet" / "partial.safetensors").write_bytes(b"half a conversion")
    (work / "tree" / "model_index.json").write_text("{}")
    return work


def stop_midway(monkeypatch) -> list[Path]:
    """`shutil.rmtree` removing one file and then meeting an open one."""
    real, met = shutil.rmtree, []

    def midway(path, *args, **kwargs):
        files = sorted(item for item in Path(path).rglob("*") if item.is_file())
        met.append(Path(path))
        if len(files) < 2:
            return real(path, *args, **kwargs)
        files[0].unlink()
        raise in_use(files[1])

    monkeypatch.setattr(shutil, "rmtree", midway)
    return met


# ---------------------------------------------------------- the happy path


def test_the_tree_goes_through_its_work_shape_and_the_lock_stays(caches, monkeypatch):
    target = caches.convert(WAI)
    stale = stale_work(target)
    lookalike = caches.converted / f"{target.name}-zzzz9999"
    lookalike.mkdir()
    (lookalike / "notes.txt").write_text("a user's own folder")
    other = caches.convert(AESTHETIC)
    unmarked = caches.older("wai-illustrious-v17-1111111111111111", None)
    kept = {key: value for key, value in caches.state().items()
            if not key.startswith("home/auradream/converted/wai-illustrious-v17-")
            or "zzzz9999" in key or "1111111111111111" in key}
    planned = delete.plan(WAI)

    renames, removed = [], []
    rename, rmtree = os.rename, shutil.rmtree

    def watch_rename(source, destination, *args, **kwargs):
        destination = Path(destination)
        # Exactly `_is_work`'s shape, made empty for the rename alone.
        assert destination.name == "tree" and single_file._is_work(destination.parent, target)
        assert list(destination.parent.iterdir()) == []
        renames.append((Path(source), destination))
        return rename(source, destination, *args, **kwargs)

    def watch_rmtree(path, *args, **kwargs):
        # Before a byte is removed the tree is no hit any more, and nothing
        # removed is anything but a work directory of the target.
        assert single_file.converted_tree(WAI) is None and not target.exists()
        assert single_file._is_work(Path(path), target)
        removed.append(Path(path).name)
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", watch_rename)
    monkeypatch.setattr(shutil, "rmtree", watch_rmtree)
    revision = model_info.cache_revision()
    outcome = delete.delete(WAI, planned.pin)

    assert [source for source, _ in renames] == [target]
    assert sorted(removed) == sorted([renames[0][1].parent.name, stale.name])
    assert outcome.changed and outcome.remaining == []
    assert outcome.planned_bytes == planned.frees_bytes > 0
    assert outcome.freed_bytes is None or outcome.freed_bytes >= 0
    assert set(outcome.answer()) == {"label", "planned_bytes", "freed_bytes", "remaining"}
    # The `.lock` stays, zero bytes; the look-alike, the unmarked older tree,
    # another entry's tree and the sentinel home byte for byte.
    lock_left_alone(caches.converted / f"{target.name}.lock")
    assert folders(caches) == {lookalike.name, other.name, unmarked.name}
    assert {key: value for key, value in caches.state().items() if key in kept} == kept
    # Nothing of it is left to plan, the inventory says so, and the facts moved.
    with pytest.raises(delete.NotOnDisk):
        delete.plan(WAI)
    snapshot = model_cache_storage.snapshot()
    assert snapshot["warnings"] == []
    row = next((row for row in snapshot["models"] if row["model_id"] == WAI), None)
    assert row is None or not row.get("converted_bytes")
    assert model_info.cache_revision() > revision
    assert not model_cache_storage._scan_lock.locked()


def test_work_directories_alone_go_and_an_older_recipe_goes_through_the_current_targets_shape(caches):
    target, _, _ = model_info._conversion_target(WAI)
    stale = stale_work(target)
    older = caches.older("wai-illustrious-v17-0000000000000000",
                         json.dumps({"version": 0, "source": single_file.WAI_SOURCE}))
    unmarked = caches.older("wai-illustrious-v17-1111111111111111", None)
    rouwei = caches.older("rouwei-old", json.dumps({"version": 0, "source": single_file.ROUWEI_SOURCE}))
    planned = delete.plan(WAI)
    assert {(part.kind, part.target) for part in planned.parts} == {("converted", target), ("converted-old", target)}
    outcome = delete.delete(WAI, planned.pin)
    assert outcome.remaining == [] and outcome.changed
    assert not stale.exists() and not older.exists()
    assert folders(caches) == {unmarked.name, rouwei.name}
    # Each lock the removal took stays: the target's and the older tree's own.
    for name in (target.name, older.name):
        lock_left_alone(caches.converted / f"{name}.lock")


def test_a_work_directory_a_load_swept_since_the_plan_changes_nothing_asked(caches):
    # Stale work is the lock holder's to remove: a load that swept it
    # between the pin's check and the lock is no changed tree (a plan read
    # before the sweep, here; a fresh plan's pin would already differ).
    target = caches.convert(WAI)
    stale = stale_work(target)
    planned = delete.plan(WAI)
    shutil.rmtree(stale)
    result = delete.Outcome(planned.label, planned.frees_bytes)
    delete.remove_converted(planned, result)
    assert result.changed and result.remaining == [] and not target.exists() and folders(caches) == set()


def test_the_delete_holds_keep_awake_and_releases_everything(caches, monkeypatch):
    caches.convert(AESTHETIC)
    holds = []
    busy = keep_awake.busy

    def counted(reason):
        holds.append((reason, model_cache_storage._scan_lock.locked()))
        return busy(reason)

    monkeypatch.setattr(keep_awake, "busy", counted)
    run(AESTHETIC)
    assert holds == [(delete.DELETING, True)]
    assert not model_cache_storage._scan_lock.locked()


# ----------------------------------------------------- refusals, unchanged


def test_a_conversion_holding_the_lock_refuses_with_nothing_changed(caches):
    # Another process's conversion, or a load sweeping: it holds the
    # target's lock from another thread here, as `test_converter_cache` does.
    from huggingface_hub.utils import WeakFileLock

    target = caches.convert(WAI)
    planned = delete.plan(WAI)
    before = converted_state(caches)
    taken, release = threading.Event(), threading.Event()

    def hold():
        with WeakFileLock(caches.converted / f"{target.name}.lock"):
            taken.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert taken.wait(10)
        with pytest.raises(delete.DeleteRefused) as refused:
            delete.delete(WAI, planned.pin, lock_timeout=0.3)
    finally:
        release.set()
        holder.join(10)
    assert refused.value.code == MODEL_DELETE_BUSY
    assert converted_state(caches) == before
    assert not model_cache_storage._scan_lock.locked()
    # Let go, the same plan goes through.
    assert delete.delete(WAI, planned.pin).remaining == []
    assert not target.exists()


def test_the_hub_part_is_stage_3s_and_refuses_before_anything(caches):
    caches.plant(WAI, only=WAI)
    caches.convert(WAI)
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        run(WAI)
    assert refused.value.code == MODEL_DELETE_HELD and "not built yet" in refused.value.detail
    assert caches.state() == before


def test_a_changed_cache_selected_or_held_entry_refuses_before_anything(caches):
    target = caches.convert(AESTHETIC)
    planned = delete.plan(AESTHETIC)
    (target / "added.json").write_text("{}")
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(AESTHETIC, planned.pin)
    assert refused.value.code == MODEL_CACHE_CHANGED
    assert caches.state() == before
    for context in ({"selected": AESTHETIC}, {"loaded": AESTHETIC}, {"saved": AESTHETIC}):
        with pytest.raises(delete.DeleteRefused) as refused:
            run(AESTHETIC, **context)
        assert refused.value.code == MODEL_SELECTED
    # An older tree from the repository Turbo and Aesthetic share, from a
    # file neither uses now: held by the plan, refused by the delete.
    caches.older("anima-preview3-0000000000000000", json.dumps({"version": 0, "source": {
        "repo_id": TURBO, "filename": "split_files/diffusion_models/anima-preview3.safetensors"}}))
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        run(AESTHETIC)
    assert refused.value.code == MODEL_DELETE_HELD
    assert caches.state() == before
    with pytest.raises(delete.NotOnDisk):
        delete.delete("nobody/nothing", planned.pin)
    with pytest.raises(delete.NotOnDisk):
        delete.delete(NETAYUME, planned.pin)
    model_cache_storage._scan_lock.acquire()
    try:
        with pytest.raises(model_cache_storage.ScanBusy):
            delete.delete(AESTHETIC, planned.pin)
    finally:
        model_cache_storage._scan_lock.release()


def test_a_tokenizer_fetch_from_the_tree_refuses_at_the_worker(caches):
    # Check began reading the converted tree between the route's guard and
    # the worker's re-plan: the worker's own check refuses before the rename.
    target = caches.convert(AESTHETIC)
    planned = delete.plan(AESTHETIC)
    assert planned.repositories
    before = caches.state()
    for source in (TokenizerSource("nobody/else", "tokenizer", tree=AESTHETIC),
                   TokenizerSource(sorted(planned.repositories)[0], "tokenizer")):
        with pytest.raises(delete.DeleteRefused) as refused:
            delete.delete(AESTHETIC, planned.pin, delete.PlanContext(pending=frozenset({source})))
        assert refused.value.code == MODEL_DELETE_BUSY
        assert caches.state() == before and target.is_dir()
        assert not model_cache_storage._scan_lock.locked()
    # A fetch from another entry's tree, from a repository it never names,
    # holds nothing here.
    other = TokenizerSource("nobody/else", "tokenizer", tree=NETAYUME)
    outcome = delete.delete(AESTHETIC, planned.pin, delete.PlanContext(pending=frozenset({other})))
    assert outcome.remaining == [] and not target.exists()


@pytest.mark.parametrize("hub_delete", [False, True], ids=["hub held", "hub built"])
def test_a_partial_download_refuses_at_the_worker(caches, monkeypatch, hub_delete):
    # Refused as a download, whether or not the Hub part would be held anyway.
    monkeypatch.setattr(delete, "HUB_DELETE", hub_delete)
    caches.plant(WAI, only=WAI)
    target = caches.convert(WAI)
    partial = caches.folder(WAI) / "blobs" / "0123abcd.incomplete"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    # The plan read it: the pin is current, so only the worker's own
    # download check stands between it and the rename.
    planned = delete.plan(WAI)
    assert planned.downloading
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(WAI, planned.pin)
    assert refused.value.code == MODEL_DOWNLOADING and "0123abcd.incomplete" in refused.value.detail
    assert caches.state() == before and target.is_dir()
    assert not model_cache_storage._scan_lock.locked()


def test_under_the_lock_a_tree_that_changed_or_appeared_is_not_taken(caches):
    # Between the plan and the lock (a stale plan here): a file added, the
    # tree swapped for a link into the sentinel home, a tree that appeared.
    target = caches.convert(WAI)
    for change in ("added", "link"):
        planned = delete.plan(WAI)
        if change == "added":
            (target / "added.json").write_text("{}")
        else:
            moved = caches.converted / "moved-away"
            target.rename(moved)
            link_folder(target, caches.sentinel / "auradream" / "converted" / "anima-turbo-v1.1-sentinel")
        before = caches.state()
        with pytest.raises(delete.DeleteRefused) as refused:
            delete.remove_converted(planned, delete.Outcome(planned.label, planned.frees_bytes))
        assert refused.value.code == MODEL_CACHE_CHANGED
        assert caches.state() == before
    target.unlink() if target.is_symlink() else os.rmdir(target)  # the junction itself, never its target
    shutil.rmtree(caches.converted / "moved-away")
    stale_work(target)
    planned = delete.plan(WAI)
    caches.convert(WAI)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.remove_converted(planned, delete.Outcome(planned.label, planned.frees_bytes))
    assert refused.value.code == MODEL_CACHE_CHANGED
    assert target.is_dir()


def test_every_path_is_reasserted_inside_the_configured_roots(caches, monkeypatch, tmp_path):
    target = caches.convert(WAI)
    planned = delete.plan(WAI)
    delete.assert_inside_roots(planned)
    part = next(part for part in planned.parts if part.kind == "converted")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    def refused(result) -> None:
        with pytest.raises(delete.DeleteRefused) as error:
            delete.assert_inside_roots(result)
        assert error.value.code == MODEL_DELETE_HELD and str(tmp_path) not in error.value.detail

    for forged in (
        {"target": elsewhere / target.name},
        {"units": [elsewhere / target.name]},
        {"units": [caches.converted / ".." / ".." / "escaped"]},
    ):
        original = {key: getattr(part, key) for key in forged}
        for key, value in forged.items():
            setattr(part, key, value)
        refused(planned)
        for key, value in original.items():
            setattr(part, key, value)
    removal = planned.removals[0]
    planned.removals[0] = delete.Removal(removal.kind, removal.root, removal.relative,
                                         caches.sentinel / "hub" / "outside", removal.size, removal.mtime_ns)
    refused(planned)
    planned.removals[0] = removal
    delete.assert_inside_roots(planned)
    # The configuration moved between the plan and the delete: another HF_HOME.
    monkeypatch.setattr(model_cache_storage, "hf_home_dir", lambda name: elsewhere / name)
    refused(planned)
    assert target.is_dir()


# ------------------------------------------------------- injected failures


@pytest.mark.parametrize("failure", ["rename", "rename and rmdir", "work directory"])
def test_a_refused_rename_changes_nothing(caches, monkeypatch, failure):
    target = caches.convert(WAI)
    stale = stale_work(target)
    planned = delete.plan(WAI)
    before = converted_state(caches)
    if failure == "work directory":
        mkdir = Path.mkdir

        def refuse_mkdir(path, *args, **kwargs):
            if path.parent == caches.converted:
                raise in_use(path)
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", refuse_mkdir)
    else:
        def refuse_rename(source, destination, *args, **kwargs):
            raise in_use(source)

        monkeypatch.setattr(os, "rename", refuse_rename)
    if failure == "rename and rmdir":
        def refuse_rmdir(path, *args, **kwargs):
            raise in_use(path)

        monkeypatch.setattr(Path, "rmdir", refuse_rmdir)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(WAI, planned.pin)
    assert refused.value.code == MODEL_FILES_IN_USE and "nothing was deleted" in refused.value.detail
    # The stale work directory too: nothing goes before the first rename.
    after = converted_state(caches)
    if failure == "rename and rmdir":
        # One empty folder of `_is_work`'s shape, which the next load sweeps.
        [left] = [name for name in after if name not in before]
        assert single_file._is_work(caches.converted / left, target)
        assert list((caches.converted / left).iterdir()) == []
        after.pop(left)
    assert after == before
    assert target.is_dir() and stale.is_dir()
    assert not model_cache_storage._scan_lock.locked()


def test_an_rmtree_that_stops_midway_leaves_only_work_directories(caches, monkeypatch):
    target = caches.convert(WAI)
    stale_work(target)
    lookalike = caches.converted / f"{target.name}-zzzz9999"
    lookalike.mkdir()
    (lookalike / "notes.txt").write_text("a user's own folder")
    planned = delete.plan(WAI)
    met = stop_midway(monkeypatch)
    outcome = delete.delete(WAI, planned.pin)
    assert len(met) == 2  # the renamed tree and the stale work directory, each tried
    assert outcome.changed
    assert [entry["kind"] for entry in outcome.remaining] == ["converted", "converted"]
    assert all(entry["bytes"] > 0 and "still open" in entry["reason"] for entry in outcome.remaining)
    text = json.dumps(outcome.answer())
    assert str(caches.home) not in text and caches.home.as_posix() not in text
    left = only_work_left(caches, target, {lookalike.name})
    assert len(left) == 2
    # The next delete finds them as the entry's and finishes the job.
    monkeypatch.setattr(shutil, "rmtree", _real_rmtree)
    again = run(WAI)
    assert again.remaining == [] and folders(caches) == {lookalike.name}


def test_an_rmtree_refused_at_once_on_stale_work_alone_changes_nothing(caches, monkeypatch):
    target, _, _ = model_info._conversion_target(WAI)
    stale = stale_work(target)
    planned = delete.plan(WAI)
    before = converted_state(caches)

    def refuse(path, *args, **kwargs):
        raise in_use(path)

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(WAI, planned.pin)
    assert refused.value.code == MODEL_FILES_IN_USE
    assert converted_state(caches) == before and stale.is_dir()


def test_a_later_failure_keeps_what_changed_and_names_what_stays(caches, monkeypatch):
    # The entry's tree goes; the older recipe's rename is then refused: a
    # 200 with `remaining` in stage 3, never a refusal saying nothing changed.
    target = caches.convert(WAI)
    older = caches.older("wai-illustrious-v17-0000000000000000",
                         json.dumps({"version": 0, "source": single_file.WAI_SOURCE}))
    planned = delete.plan(WAI)
    rename = os.rename

    def refuse_older(source, destination, *args, **kwargs):
        if Path(source) == older:
            raise in_use(source)
        return rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", refuse_older)
    outcome = delete.delete(WAI, planned.pin)
    assert outcome.changed and not target.exists()
    older_part = next(part for part in planned.parts if part.kind == "converted-old")
    assert outcome.remaining == [{"kind": "converted-old", "bytes": older_part.bytes,
                                  "reason": "A file in it is open, so it could not be moved aside."}]
    assert older.is_dir() and (older / single_file.CONVERSION_MARKER).is_file()
    only_work_left(caches, target, {older.name})


def test_an_older_trees_leftover_is_swept_by_the_entrys_next_load(caches, monkeypatch):
    older = caches.older("wai-illustrious-v17-0000000000000000",
                         json.dumps({"version": 0, "source": single_file.WAI_SOURCE}))
    (older / "unet").mkdir()
    (older / "unet" / "weights.safetensors").write_bytes(b"older weights")
    target, _, _ = model_info._conversion_target(WAI)
    planned = delete.plan(WAI)
    stop_midway(monkeypatch)
    outcome = delete.delete(WAI, planned.pin)
    assert outcome.changed and [entry["kind"] for entry in outcome.remaining] == ["converted-old"]
    [left] = only_work_left(caches, target, set())
    monkeypatch.setattr(shutil, "rmtree", _real_rmtree)
    calls = _fake_conversion(monkeypatch, single_file)
    assert Path(single_file.resolve_model(WAI, "fp8")) == target
    assert not left.exists() and len(calls) == 1
    assert folders(caches) == {target.name}


# --------------------------------------------- a later load converts cleanly


def _fake_conversion(monkeypatch, module) -> list[Path]:
    calls: list[Path] = []

    def convert(path, *args):
        calls.append(path)
        (path / "transformer").mkdir(parents=True)
        (path / "model_index.json").write_text("{}")
        (path / "transformer" / "weights.safetensors").write_bytes(b"converted weights")

    monkeypatch.setattr(module, "_convert", convert)
    return calls


LOADERS = [(single_file, WAI), (anima_loading, TURBO), (anima_loading, AESTHETIC),
           (netayume_loading, NETAYUME), (newbie_loading, NEWBIE)]


@pytest.mark.parametrize("midway", [False, True], ids=["whole", "stopped midway"])
@pytest.mark.parametrize("module, model", LOADERS, ids=[model for _, model in LOADERS])
def test_a_later_load_converts_cleanly_in_the_scratch_cache(caches, monkeypatch, module, model, midway):
    calls = _fake_conversion(monkeypatch, module)
    first = Path(module.resolve_model(model, "fp8"))
    assert first.parent == caches.converted and module.converted_tree(model) == str(first)
    planned = delete.plan(model)
    assert [part.kind for part in planned.parts] == ["converted"]
    if midway:
        stop_midway(monkeypatch)
    outcome = delete.delete(model, planned.pin)
    monkeypatch.setattr(shutil, "rmtree", _real_rmtree)
    assert module.converted_tree(model) is None
    if midway:
        assert outcome.remaining and only_work_left(caches, first, set())
    else:
        assert outcome.remaining == [] and folders(caches) == set()
    again = Path(module.resolve_model(model, "fp8"))
    assert again == first and len(calls) == 2
    assert folders(caches) == {first.name}
    assert module.converted_tree(model) == str(first)
    # And the delete plans it again as the entry's, whole.
    assert [part.kind for part in delete.plan(model).parts] == ["converted"]
