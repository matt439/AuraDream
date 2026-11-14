"""The checkpoint delete's Hub removal (`GEN#72` stage 3), on scratch caches.

docs/maintainer/plan-model-delete.md, "How files are removed > A repository
E owns outright", is the spec: a repository only the entry's group names
(its own, its Nunchaku lite ones, a cached lite-infer alias) is renamed out
of the Hub cache into the sibling trash (`<hub>.auradream-deleting/<8 hex>/`)
- every part, converted trees included, renamed before any is removed -
then removed from there. A refused rename, whichever part's, puts back
what moved and changes nothing (`model_files_in_use`); what an open file
keeps in the trash is the outcome's `remaining`, named by the inventory as
a warning and swept by the next delete. A removal inside a
repository other entries use is stage 4's, file by file:
tests/test_model_cache_delete_shared.py.
"""
from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

import pytest

from backend import model_cache_delete as delete, model_cache_storage, model_info
from backend.backends import single_file
from backend.refusal_codes import MODEL_CACHE_CHANGED, MODEL_DELETE_HELD, MODEL_DOWNLOADING, MODEL_FILES_IN_USE
from tests.model_cache_fixtures import LAYOUTS, link_folder, scratch_caches

ERNIE = "baidu/ERNIE-Image-Turbo"
KREA2 = "krea/Krea-2-Turbo"
WAI, ILLUSTRIOUS = single_file.WAI_ID, single_file.ILLUSTRIOUS_ID
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
SHARING_VIOLATION = 32
OLD = "8" * 40


@pytest.fixture(params=LAYOUTS)
def caches(request, tmp_path, monkeypatch):
    return scratch_caches(tmp_path, monkeypatch, request.param)


@pytest.fixture
def copy_caches(tmp_path, monkeypatch):
    return scratch_caches(tmp_path, monkeypatch, "copy")


def in_use(path) -> PermissionError:
    return PermissionError(13, "The process cannot access the file because it is being used by another process",
                           str(path), SHARING_VIOLATION)


def run(model: str, **context) -> delete.Outcome:
    planned = delete.plan(model, delete.PlanContext(**context))
    return delete.delete(model, planned.pin, delete.PlanContext(**context))


def trash(caches) -> Path:
    return model_cache_storage.trash_dir(caches.hub.resolve())


def krea2_group(caches) -> tuple[str, str]:
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None
    nvfp4, alias = entry.lite_repo("nvfp4"), entry.lite_alias("int4")
    assert nvfp4 and alias
    caches.plant(KREA2)
    caches.plant(nvfp4)
    caches.put(alias, "6" * 40, "tokenizer/tokenizer_config.json")
    return nvfp4, alias


def without(state: dict, *stems: str) -> dict:
    return {key: value for key, value in state.items()
            if not any(key == f"home/hub/{stem}" or key.startswith(f"home/hub/{stem}/") for stem in stems)}


def stem(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


# ---------------------------------------------------------- the happy path


def test_an_owned_repository_goes_whole_through_the_trash(caches, monkeypatch):
    _, commit, _ = caches.recipe(ERNIE)[0]
    caches.plant(ERNIE)
    caches.ref(ERNIE, commit)
    caches.put(ERNIE, commit, "pe/model.safetensors")  # unselected: Expand's rewriter
    caches.put(ERNIE, OLD, "transformer/config.json", b"an older download")
    missing = caches.folder(ERNIE) / ".no_exist" / commit / "model.fp16.safetensors"
    missing.parent.mkdir(parents=True)
    missing.write_bytes(b"")
    caches.plant(KREA2)  # another entry's repository: untouched
    caches.put("someone/unlisted-model", "4" * 40, "weights.safetensors")  # no entry's
    kept = without(caches.state(), stem(ERNIE))
    planned = delete.plan(ERNIE)
    renames, removed = [], []
    rename, rmtree = os.rename, shutil.rmtree

    def watch_rename(source, destination, *args, **kwargs):
        destination = Path(destination)
        # `<hub>.auradream-deleting/<8 hex>/models--org--name`, the bin made empty for it.
        assert destination.parent.parent == trash(caches)
        assert len(destination.parent.name) == 8 and int(destination.parent.name, 16) >= 0
        assert destination.name == Path(source).name and list(destination.parent.iterdir()) == []
        renames.append(Path(source))
        return rename(source, destination, *args, **kwargs)

    def watch_rmtree(path, *args, **kwargs):
        removed.append(Path(path))
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", watch_rename)
    monkeypatch.setattr(shutil, "rmtree", watch_rmtree)
    outcome = delete.delete(ERNIE, planned.pin)
    assert renames == [caches.folder(ERNIE).resolve()] or renames == [caches.folder(ERNIE)]
    assert [path.parent for path in removed] == [trash(caches)]
    assert outcome.remaining == [] and outcome.changed and outcome.planned_bytes == planned.frees_bytes > 0
    assert outcome.freed_bytes is not None and outcome.freed_bytes >= 0
    assert not os.path.lexists(caches.folder(ERNIE))
    # The trash goes with its last folder; everything else byte for byte.
    assert not os.path.lexists(trash(caches))
    assert caches.state() == kept
    answer = model_cache_storage.snapshot()
    row = next(row for row in answer["models"] if row["model_id"] == ERNIE)
    assert (row["revision_bytes"], row["other_revision_bytes"], row["converted_bytes"]) == (0, 0, 0)
    assert answer["warnings"] == []
    assert not model_cache_storage._scan_lock.locked()


def test_the_lite_repositories_and_alias_all_move_before_any_is_removed(caches, monkeypatch):
    nvfp4, alias = krea2_group(caches)
    planned = delete.plan(KREA2)
    events = []
    rename, rmtree = os.rename, shutil.rmtree

    def watch_rename(source, destination, *args, **kwargs):
        events.append(("rename", Path(source).name))
        return rename(source, destination, *args, **kwargs)

    def watch_rmtree(path, *args, **kwargs):
        events.append(("rmtree", Path(path).parent.name))
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", watch_rename)
    monkeypatch.setattr(shutil, "rmtree", watch_rmtree)
    outcome = delete.delete(KREA2, planned.pin)
    assert outcome.remaining == []
    assert [kind for kind, _ in events] == ["rename"] * 3 + ["rmtree"] * 3
    assert {name for kind, name in events if kind == "rename"} == {stem(KREA2), stem(nvfp4), stem(alias)}
    for repo in (KREA2, nvfp4, alias):
        assert not os.path.lexists(caches.folder(repo))
    assert not os.path.lexists(trash(caches))


def test_a_converted_copy_and_its_source_repository_go_together(copy_caches, monkeypatch):
    caches = copy_caches
    caches.plant(WAI, only=WAI)
    target = caches.convert(WAI)
    events = []
    rename, rmtree = os.rename, shutil.rmtree

    def watch_rename(source, destination, *args, **kwargs):
        events.append(("rename", Path(source).name))
        return rename(source, destination, *args, **kwargs)

    def watch_rmtree(path, *args, **kwargs):
        events.append(("rmtree", Path(path).name))
        return rmtree(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", watch_rename)
    monkeypatch.setattr(shutil, "rmtree", watch_rmtree)
    outcome = run(WAI)
    assert outcome.remaining == [] and outcome.changed
    # Both renamed before either is removed: the converted tree first.
    assert [kind for kind, _ in events] == ["rename", "rename", "rmtree", "rmtree"]
    assert [name for kind, name in events if kind == "rename"] == [target.name, stem(WAI)]
    assert not os.path.lexists(caches.folder(WAI)) and not target.exists()
    assert {path.name for path in caches.converted.iterdir() if path.is_dir()} == set()


# ------------------------------------------------------- shared parts kept


def test_files_a_present_sharer_keeps_stay_and_the_rest_goes(caches):
    # With WAI on disk the shared SDXL base files are kept, not taken: the
    # plan removes nothing inside a shared repository, and its delete leaves
    # SDXL base byte for byte.
    caches.plant(ILLUSTRIOUS)
    caches.plant(WAI)
    planned = delete.plan(ILLUSTRIOUS)
    assert planned.held is None and planned.kept_shared
    assert all(part.whole and not part.shared_repository for part in planned.parts)
    kept = without(caches.state(), stem(ILLUSTRIOUS))
    assert delete.delete(ILLUSTRIOUS, planned.pin).remaining == []
    assert caches.state() == kept and caches.folder(SDXL).is_dir()


# ------------------------------------------------------- injected failures


@pytest.mark.parametrize("failure", ["in use", "another drive", "bin"])
def test_a_refused_first_rename_changes_nothing(copy_caches, monkeypatch, failure):
    caches = copy_caches
    caches.plant(ERNIE)
    planned = delete.plan(ERNIE)
    before = caches.state()
    if failure == "bin":
        mkdir = Path.mkdir

        def refuse_mkdir(path, *args, **kwargs):
            if path.parent == trash(caches):
                raise in_use(path)
            return mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", refuse_mkdir)
    else:
        def refuse_rename(source, destination, *args, **kwargs):
            if failure == "another drive":
                raise OSError(errno.EXDEV, "Invalid cross-device link", str(source))
            raise in_use(source)

        monkeypatch.setattr(os, "rename", refuse_rename)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(ERNIE, planned.pin)
    assert refused.value.code == MODEL_FILES_IN_USE and "nothing was deleted" in refused.value.detail
    if failure == "another drive":
        assert "another drive" in refused.value.detail
    assert str(caches.home) not in refused.value.detail
    assert caches.state() == before
    assert not os.path.lexists(trash(caches))
    assert not model_cache_storage._scan_lock.locked()


def test_a_later_refused_rename_puts_back_what_moved(copy_caches, monkeypatch):
    # The group's last repository is refused after the others moved: they
    # come back, and the delete refuses with nothing changed.
    caches = copy_caches
    krea2_group(caches)
    planned = delete.plan(KREA2)
    last = [part for part in planned.parts if part.kind in ("hub", "lite")][-1]
    assert last.repo is not None
    before = caches.state()
    rename, moved = os.rename, []

    def refuse_last(source, destination, *args, **kwargs):
        if Path(source).name == stem(last.repo):
            raise in_use(source)
        moved.append(Path(source).name)
        return rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", refuse_last)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(KREA2, planned.pin)
    assert refused.value.code == MODEL_FILES_IN_USE and "nothing was deleted" in refused.value.detail
    assert len(moved) == 4  # two moved out, and back
    assert caches.state() == before
    assert not os.path.lexists(trash(caches))
    assert not model_cache_storage._scan_lock.locked()
    monkeypatch.setattr(os, "rename", rename)
    assert delete.delete(KREA2, planned.pin).remaining == []


def without_locks(state: dict) -> dict:
    """The zero-byte `.lock` a converted part's lock leaves (by design)."""
    return {key: value for key, value in state.items() if not key.endswith(".lock")}


@pytest.mark.parametrize("failure", ["in use", "download"])
def test_a_refused_hub_part_puts_the_converted_copy_back(copy_caches, monkeypatch, failure):
    # WAI's converted copy goes aside first; then its source repository is
    # refused (a file open in another program, or a download that started
    # since the plan). The copy comes back: nothing removed, as the docs say.
    caches = copy_caches
    caches.plant(WAI, only=WAI)
    target = caches.convert(WAI)
    planned = delete.plan(WAI)
    assert sorted(part.kind for part in planned.parts) == ["converted", "hub"]
    partial = caches.folder(WAI) / "blobs" / "0123abcd.incomplete"
    before = without_locks(caches.state())
    rename, moved = os.rename, []

    def refuse_hub(source, destination, *args, **kwargs):
        if Path(source).name == stem(WAI) and failure == "in use":
            raise in_use(source)
        moved.append(Path(source).name)
        rename(source, destination, *args, **kwargs)
        if Path(source) == target and failure == "download":
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"half a file")

    monkeypatch.setattr(os, "rename", refuse_hub)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(WAI, planned.pin)
    assert refused.value.code == (MODEL_FILES_IN_USE if failure == "in use" else MODEL_DOWNLOADING)
    assert moved[:2] == [target.name, "tree"]  # aside, and back
    if failure == "download":
        partial.unlink()
        if not any(partial.parent.iterdir()):
            partial.parent.rmdir()  # made for the partial alone
    assert without_locks(caches.state()) == before
    assert target.is_dir() and caches.folder(WAI).is_dir()
    assert not os.path.lexists(trash(caches))
    assert not model_cache_storage._scan_lock.locked()
    monkeypatch.setattr(os, "rename", rename)
    assert delete.delete(WAI, planned.pin).remaining == []
    assert not target.exists() and not os.path.lexists(caches.folder(WAI))


def test_a_rename_that_will_not_go_back_is_finished_and_says_what_stays(copy_caches, monkeypatch):
    # The double fault: the Hub rename refused, and the converted copy will
    # not go back either. It is out of the loaders' sight already, so it is
    # removed, and the answer names the repository that stays.
    caches = copy_caches
    caches.plant(WAI, only=WAI)
    target = caches.convert(WAI)
    planned = delete.plan(WAI)
    rename = os.rename

    def refuse(source, destination, *args, **kwargs):
        if Path(source).name in (stem(WAI), "tree"):
            raise in_use(source)
        return rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "rename", refuse)
    outcome = delete.delete(WAI, planned.pin)
    hub = next(part for part in planned.parts if part.kind == "hub")
    assert outcome.changed
    assert outcome.remaining == [{"kind": "hub", "bytes": hub.bytes,
                                  "reason": "A file in it is open, so it could not be moved out of the cache."}]
    assert not target.exists() and caches.folder(WAI).is_dir()
    assert {path.name for path in caches.converted.iterdir() if path.is_dir()} == set()
    assert not os.path.lexists(trash(caches))


def test_an_open_file_in_the_trash_is_named_and_swept_by_the_next_delete(copy_caches, monkeypatch):
    caches = copy_caches
    caches.plant(ERNIE)
    caches.plant(KREA2)
    planned = delete.plan(ERNIE)
    rmtree = shutil.rmtree

    def stop_midway(path, *args, **kwargs):
        files = sorted(item for item in Path(path).rglob("*") if item.is_file())
        files[0].unlink()
        raise in_use(files[1])

    monkeypatch.setattr(shutil, "rmtree", stop_midway)
    outcome = delete.delete(ERNIE, planned.pin)
    [left] = outcome.remaining
    assert left["kind"] == "hub" and 0 < left["bytes"] < planned.frees_bytes
    assert "out of the cache" in left["reason"]
    # Out of the Hub cache already: the inventory no longer counts it, and
    # says what is left beside it, by the trash's name alone.
    assert not os.path.lexists(caches.folder(ERNIE))
    answer = model_cache_storage.snapshot()
    row = next(row for row in answer["models"] if row["model_id"] == ERNIE)
    assert row["revision_bytes"] == 0
    [warning] = answer["warnings"]
    assert "hub.auradream-deleting" in warning and str(caches.home) not in warning
    # The next delete, of another entry, sweeps it first.
    monkeypatch.setattr(shutil, "rmtree", rmtree)
    assert run(KREA2).remaining == []
    assert not os.path.lexists(trash(caches))
    assert model_cache_storage.snapshot()["warnings"] == []


def test_the_sweep_takes_only_its_own_folders(copy_caches):
    caches = copy_caches
    bin_ = trash(caches) / "0123abcd"
    (bin_ / "models--x--y" / "snapshots").mkdir(parents=True)
    (bin_ / "models--x--y" / "snapshots" / "file").write_bytes(b"left")
    stranger = trash(caches) / "not-ours"
    stranger.mkdir()
    (stranger / "keep.txt").write_text("someone's")
    assert delete.sweep_trash(trash(caches)) == ["not-ours"]
    assert not bin_.exists() and (stranger / "keep.txt").read_text() == "someone's"


def test_a_linked_trash_refuses_before_anything(copy_caches, tmp_path):
    caches = copy_caches
    caches.plant(ERNIE)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link_folder(trash(caches), elsewhere)
    planned = delete.plan(ERNIE)
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(ERNIE, planned.pin)
    assert refused.value.code == MODEL_DELETE_HELD
    assert caches.state() == before and list(elsewhere.iterdir()) == []


def test_a_repository_gone_before_its_rename_is_a_changed_cache(copy_caches, monkeypatch):
    caches = copy_caches
    caches.plant(ERNIE)
    planned = delete.plan(ERNIE)
    result = delete._plan(ERNIE, delete.PlanContext(), None)
    shutil.rmtree(caches.folder(ERNIE))
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.remove_hub(result, delete.Outcome(planned.label, planned.frees_bytes))
    assert refused.value.code == MODEL_CACHE_CHANGED
    assert not os.path.lexists(trash(caches))


def test_a_plan_taking_ernies_rewriter_says_so(copy_caches):
    from backend.backends import loading

    caches = copy_caches
    assert delete.PROMPT_EXPANDER_WEIGHTS == loading.PROMPT_EXPANDER_WEIGHTS
    _, commit, _ = caches.recipe(ERNIE)[0]
    caches.plant(ERNIE)
    assert delete.plan(ERNIE).answer()["removes_expander"] is False
    caches.put(ERNIE, commit, delete.PROMPT_EXPANDER_WEIGHTS)
    assert delete.plan(ERNIE).answer()["removes_expander"] is True
    caches.plant(KREA2)
    assert delete.plan(KREA2).answer()["removes_expander"] is False
