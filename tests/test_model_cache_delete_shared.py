"""The checkpoint delete inside shared repositories (`GEN#72` stage 4), on
scratch caches in both layouts.

docs/maintainer/plan-model-delete.md, "Files inside a repository others use"
and "Shared files", is the spec: Anima Turbo and Aesthetic (two commits of one
repository), the Anima base three entries select, SDXL base's configs and
tokenizers for Illustrious, WAI and RouWei. A file another entry selects too
stays while that sharer is present and goes with the last holder. For each
revision the delete empties its refs go first, then each snapshot entry,
then each blob no surviving entry resolves to; emptied folders bottom-up,
then `.no_exist/<commit>`. A refusal before the first unlink changes
nothing, the converted and whole-repository renames included; after it, what
stays is `remaining` and the cache is always "files missing from a pinned
revision", never a ref to a missing snapshot, and sibling revisions are
untouched byte for byte.

The symlink layout runs where the test can make links (Linux and WSL; on
Windows only with the privilege).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend import model_cache_delete as delete, model_cache_storage
from backend.backends import single_file
from backend.refusal_codes import MODEL_CACHE_CHANGED, MODEL_DELETE_HELD, MODEL_DOWNLOADING, MODEL_FILES_IN_USE
from tests.model_cache_fixtures import LAYOUTS, scratch_caches

TURBO = "circlestone-labs/Anima"
AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
PREVIEW = "Gazingstars123/Anima-2.9B"
ANIMA_BASE = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ILLUSTRIOUS, WAI, ROUWEI = single_file.ILLUSTRIOUS_ID, single_file.WAI_ID, single_file.ROUWEI_ID
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
OLD = "8" * 40
SHARING_VIOLATION = 32


@pytest.fixture(params=LAYOUTS)
def caches(request, tmp_path, monkeypatch):
    return scratch_caches(tmp_path, monkeypatch, request.param)


def run(model: str, **context) -> delete.Outcome:
    planned = delete.plan(model, delete.PlanContext(**context))
    return delete.delete(model, planned.pin, delete.PlanContext(**context))


def in_use(path) -> PermissionError:
    return PermissionError(13, "The process cannot access the file because it is being used by another process",
                           str(path), SHARING_VIOLATION)


def stem(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def commit_of(caches, model: str, repo: str) -> str:
    return next(commit for source, commit, _ in caches.recipe(model) if source == repo)


def without_locks(state: dict) -> dict:
    """The zero-byte `.lock` a converted part's lock may leave (by design)."""
    return {key: value for key, value in state.items() if not key.endswith(".lock")}


def under(state: dict, *prefixes: str) -> dict:
    return {key: value for key, value in state.items() if any(key.startswith(prefix) for prefix in prefixes)}


def readable(caches) -> None:
    """The Hub's own scan reads the cache with no corrupted repository, and
    every ref names a snapshot that exists."""
    from huggingface_hub import scan_cache_dir

    info = scan_cache_dir(caches.hub)
    assert list(info.warnings) == [], info.warnings
    for folder in caches.hub.glob("models--*"):
        refs = folder / "refs"
        for ref in (path for path in refs.rglob("*") if path.is_file()) if refs.is_dir() else ():
            assert (folder / "snapshots" / ref.read_text().strip()).is_dir(), (folder.name, ref.name)
    assert model_cache_storage.snapshot()["warnings"] == []


def notes(caches, repo: str, commit: str) -> Path:
    """A `.no_exist/<commit>` note, as the Hub client writes for a file a
    revision lacks."""
    note = caches.folder(repo) / ".no_exist" / commit / "model.fp16.safetensors"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_bytes(b"")
    return note


# ------------------------------------------------- Anima: two commits, one repo


def test_aesthetic_goes_file_by_file_and_turbos_revision_stays_byte_for_byte(caches):
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    turbo, aesthetic = commit_of(caches, TURBO, TURBO), commit_of(caches, AESTHETIC, TURBO)
    caches.ref(TURBO, turbo)
    caches.ref(TURBO, aesthetic, "aesthetic-v1.1")
    caches.ref(TURBO, aesthetic, "refs/pr/1")
    kept_note, gone_note = notes(caches, TURBO, turbo), notes(caches, TURBO, aesthetic)
    before = caches.state()
    planned = delete.plan(AESTHETIC)
    [part] = planned.parts
    assert part.shared_repository and part.emptied and part.revision == aesthetic
    assert sorted(ref.relative_to(caches.folder(TURBO)).as_posix() for ref in part.refs) == [
        "refs/aesthetic-v1.1", "refs/refs/pr/1"]
    assert planned.kept_shared[0]["with"] == ["Anima Turbo v1.1"] and planned.held is None
    outcome = delete.delete(AESTHETIC, planned.pin)
    assert outcome.remaining == [] and outcome.changed and outcome.planned_bytes == part.bytes > 0
    # Aesthetic's revision, its refs and notes are gone; nothing else changed.
    folder = f"home/hub/{stem(TURBO)}"
    gone = {key for key in before if key.startswith((f"{folder}/snapshots/{aesthetic}", f"{folder}/.no_exist/{aesthetic}",
                                                     f"{folder}/refs/aesthetic", f"{folder}/refs/refs"))}
    if caches.layout == "symlink":
        blob = os.path.basename(os.readlink(caches.folder(TURBO) / "snapshots" / turbo / caches.own_file(TURBO)[2]))
        gone |= {key for key in before if key.startswith(f"{folder}/blobs/") and not key.endswith(blob)}
    after = caches.state()
    assert {key: value for key, value in before.items() if key not in gone} == after
    assert not gone_note.exists() and kept_note.exists()
    assert (caches.folder(TURBO) / "refs" / "main").read_text() == turbo
    readable(caches)
    rows = {row["model_id"]: row for row in model_cache_storage.snapshot()["models"]}
    assert rows[AESTHETIC]["exclusive_bytes"] == 0 and rows[TURBO]["exclusive_bytes"] > 0


def test_a_blob_turbos_revision_still_names_stays(caches):
    # On a symlink cache one blob can serve both commits: only Aesthetic's
    # link goes, and Turbo's file still reads its bytes.
    same = b"the same bytes at two commits"
    for model in (TURBO, AESTHETIC):
        repo, commit, name = caches.own_file(model)
        caches.put(repo, commit, name, same)
    turbo = commit_of(caches, TURBO, TURBO)
    before = under(caches.state(), f"home/hub/{stem(TURBO)}/snapshots/{turbo}", f"home/hub/{stem(TURBO)}/blobs")
    assert run(AESTHETIC).remaining == []
    assert under(caches.state(), f"home/hub/{stem(TURBO)}/snapshots/{turbo}",
                 f"home/hub/{stem(TURBO)}/blobs") == before
    assert (caches.folder(TURBO) / "snapshots" / turbo / caches.own_file(TURBO)[2]).read_bytes() == same
    readable(caches)


def test_the_anima_three_in_turn_the_last_takes_the_base(caches):
    for model in (TURBO, AESTHETIC, PREVIEW):
        caches.plant(model)
    caches.ref(ANIMA_BASE, commit_of(caches, TURBO, ANIMA_BASE))
    base = under(caches.state(), f"home/hub/{stem(ANIMA_BASE)}")
    for model in (TURBO, AESTHETIC):
        assert run(model).remaining == []
        assert under(caches.state(), f"home/hub/{stem(ANIMA_BASE)}") == base
        readable(caches)
    # Both commits gone: the repository's folder goes with its last file.
    assert not os.path.lexists(caches.folder(TURBO))
    planned = delete.plan(PREVIEW)
    assert planned.kept_shared == [] and planned.also_taken_shared[0]["with"] == ["Anima Aesthetic v1.1",
                                                                                  "Anima Turbo v1.1"]
    assert run(PREVIEW).remaining == []
    assert not os.path.lexists(caches.folder(ANIMA_BASE)) and not os.path.lexists(caches.folder(PREVIEW))
    readable(caches)


# ------------------------------------------------- SDXL base: the last holder


def test_the_sdxl_three_in_turn_keep_the_base_until_the_last(caches):
    for model in (ILLUSTRIOUS, WAI, ROUWEI):
        caches.plant(model)
    caches.ref(SDXL, commit_of(caches, ILLUSTRIOUS, SDXL))
    base = under(caches.state(), f"home/hub/{stem(SDXL)}")
    for model in (WAI, ROUWEI):
        assert run(model).remaining == []
        assert under(caches.state(), f"home/hub/{stem(SDXL)}") == base
    planned = delete.plan(ILLUSTRIOUS)
    assert planned.also_taken_shared and planned.held is None
    assert run(ILLUSTRIOUS).remaining == []
    assert not os.path.lexists(caches.folder(SDXL))
    readable(caches)


def test_a_present_sharer_saved_or_selected_keeps_the_base(caches):
    caches.plant(ILLUSTRIOUS)
    base = under(caches.state(), f"home/hub/{stem(SDXL)}")
    # Illustrious selected, WAI the one to delete: nothing of WAI's on disk
    # but the base, kept for the selected sharer: nothing of its own to go.
    for context in ({"selected": WAI}, {"saved": WAI}, {"loaded": WAI}):
        planned = delete.plan(ILLUSTRIOUS, delete.PlanContext(**context))
        assert planned.kept_shared and not any(part.shared_repository for part in planned.parts)
        assert planned.answer()["selected"] is False
    assert run(ILLUSTRIOUS, selected=WAI).remaining == []
    assert under(caches.state(), f"home/hub/{stem(SDXL)}") == base
    readable(caches)


def test_an_unselected_file_keeps_the_revision_its_ref_and_notes(caches):
    caches.plant(ILLUSTRIOUS)
    commit = commit_of(caches, ILLUSTRIOUS, SDXL)
    caches.ref(SDXL, commit)
    caches.put(SDXL, commit, "README.md")  # no recipe selects it: it stays
    note = notes(caches, SDXL, commit)
    planned = delete.plan(ILLUSTRIOUS)
    sdxl = next(part for part in planned.parts if part.repo == SDXL)
    assert not sdxl.emptied and sdxl.refs == []
    assert run(ILLUSTRIOUS).remaining == []
    snapshot = caches.folder(SDXL) / "snapshots" / commit
    assert sorted(path.name for path in snapshot.iterdir()) == ["README.md"]  # emptied folders gone
    assert (caches.folder(SDXL) / "refs" / "main").read_text() == commit and note.exists()
    readable(caches)


def test_a_sibling_revision_and_its_ref_stay_byte_for_byte(caches):
    caches.plant(ILLUSTRIOUS)
    commit = commit_of(caches, ILLUSTRIOUS, SDXL)
    caches.ref(SDXL, commit)
    caches.put(SDXL, OLD, "model_index.json", b"an older download")
    caches.ref(SDXL, OLD, "old")
    sibling = under(caches.state(), f"home/hub/{stem(SDXL)}/snapshots/{OLD}", f"home/hub/{stem(SDXL)}/refs/old")
    assert run(ILLUSTRIOUS).remaining == []
    assert not (caches.folder(SDXL) / "snapshots" / commit).exists()
    assert not (caches.folder(SDXL) / "refs" / "main").exists()
    assert under(caches.state(), f"home/hub/{stem(SDXL)}/snapshots/{OLD}",
                 f"home/hub/{stem(SDXL)}/refs/old") == sibling
    readable(caches)


# --------------------------------------------------------------- still held


def test_a_shared_xet_store_still_holds_a_shared_part(caches):
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    (caches.hub / "blobs").mkdir()
    (caches.hub / "blobs" / delete.SHARED_STORE_MARKER).write_text("1\n")
    before = caches.state()
    planned = delete.plan(AESTHETIC)
    assert planned.held is not None and "shared Xet store" in planned.held["detail"]
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(AESTHETIC, planned.pin)
    assert refused.value.code == MODEL_DELETE_HELD and caches.state() == before


def test_a_ref_added_after_the_plan_changes_its_pin(caches):
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    planned = delete.plan(AESTHETIC)
    caches.ref(TURBO, commit_of(caches, AESTHETIC, TURBO), "late")
    before = caches.state()
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(AESTHETIC, planned.pin)
    assert refused.value.code == MODEL_CACHE_CHANGED and caches.state() == before


# ------------------------------------------------------- injected failures


def _illustrious_with_everything(caches) -> tuple[str, dict]:
    """Illustrious alone, the last holder: its own repository (whole, the
    trash), its converted copy (renamed aside) and SDXL base's files (file by
    file, an emptied revision with a ref), beside a sibling SDXL revision."""
    caches.plant(ILLUSTRIOUS)
    caches.convert(ILLUSTRIOUS)
    commit = commit_of(caches, ILLUSTRIOUS, SDXL)
    caches.ref(SDXL, commit)
    notes(caches, SDXL, commit)
    caches.put(SDXL, OLD, "model_index.json", b"an older download")
    caches.ref(SDXL, OLD, "old")
    sibling = under(caches.state(), f"home/hub/{stem(SDXL)}/snapshots/{OLD}", f"home/hub/{stem(SDXL)}/refs/old")
    return commit, sibling


@pytest.mark.parametrize("failure", ["a ref", "a download", "a changed file", "a new file"])
def test_a_change_found_before_the_first_unlink_puts_every_rename_back(caches, monkeypatch, failure):
    commit, _ = _illustrious_with_everything(caches)
    planned = delete.plan(ILLUSTRIOUS)
    before = without_locks(caches.state())
    snapshot = caches.folder(SDXL) / "snapshots" / commit
    rename, renamed = os.rename, []

    def change_after_the_renames(source, destination, *args, **kwargs):
        rename(source, destination, *args, **kwargs)
        renamed.append(Path(source).name)
        if Path(source).name != stem(ILLUSTRIOUS) or len(renamed) > 2:
            return
        # The whole repository has gone to the trash: then the change.
        if failure == "a ref":
            caches.ref(SDXL, commit, "late")
        elif failure == "a download":
            (caches.folder(SDXL) / "blobs").mkdir(exist_ok=True)
            (caches.folder(SDXL) / "blobs" / "0123abcd.incomplete").write_bytes(b"half")
        elif failure == "a changed file":
            entry = snapshot / "model_index.json"
            stamp = entry.lstat()
            os.utime(entry, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000_000),
                     **({"follow_symlinks": False} if entry.is_symlink() else {}))
        else:
            caches.put(SDXL, commit, "unet/diffusion_pytorch_model.fp16.safetensors")

    monkeypatch.setattr(os, "rename", change_after_the_renames)
    with pytest.raises(delete.DeleteRefused) as refused:
        delete.delete(ILLUSTRIOUS, planned.pin)
    assert refused.value.code == (MODEL_DOWNLOADING if failure == "a download" else MODEL_CACHE_CHANGED)
    # Renamed aside (the converted copy, the repository) and back.
    assert stem(ILLUSTRIOUS) in renamed and len(renamed) == 4
    after = without_locks(caches.state())
    changed = {key for key in set(before) ^ set(after)} | {key for key in before if key in after and before[key] != after[key]}
    assert all(f"{stem(SDXL)}/" in key for key in changed), changed  # only the injected change
    assert caches.folder(ILLUSTRIOUS).is_dir() and (caches.folder(SDXL) / "refs" / "main").exists()
    assert not os.path.lexists(model_cache_storage.trash_dir(caches.hub.resolve()))
    assert not model_cache_storage._scan_lock.locked()


def _unlinks(planned: delete.Plan) -> dict[str, int]:
    """The file-level step's unlinks in order: the ref, then each entry, then
    each blob (a symlink cache's)."""
    [sdxl] = [part for part in planned.parts if part.shared_repository]
    refs, entries, blobs = len(sdxl.refs), len(sdxl.entries), len(sdxl.blobs)
    positions = {"the ref": 0, "the first entry": refs, "an entry midway": refs + entries // 2,
                 "the last entry": refs + entries - 1}
    if blobs:
        positions.update({"the first blob": refs + entries, "the last blob": refs + entries + blobs - 1})
    return positions


@pytest.mark.parametrize("where", ["the ref", "the first entry", "an entry midway", "the last entry",
                                   "the first blob", "the last blob"])
def test_an_unlink_refused_leaves_no_ref_to_a_missing_snapshot(caches, monkeypatch, where):
    commit, sibling = _illustrious_with_everything(caches)
    planned = delete.plan(ILLUSTRIOUS)
    positions = _unlinks(planned)
    if where not in positions:
        pytest.skip("a plain-copy cache has no blobs behind its snapshots")
    before = without_locks(caches.state())
    unlink, calls = os.unlink, []
    hub = os.path.normcase(str(caches.hub.resolve()))

    def refuse_one(path, *args, **kwargs):
        if os.path.normcase(os.path.abspath(path)).startswith(hub + os.sep) and ".no_exist" not in str(path):
            calls.append(Path(path))
            if len(calls) - 1 == positions[where]:
                raise in_use(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse_one)
    if where == "the ref":
        # The first unlink: nothing unlinked yet, so every rename goes back.
        with pytest.raises(delete.DeleteRefused) as refused:
            delete.delete(ILLUSTRIOUS, planned.pin)
        assert refused.value.code == MODEL_FILES_IN_USE and "nothing was deleted" in refused.value.detail
        assert calls[0].parent.name == "refs"
        assert without_locks(caches.state()) == before
        assert not os.path.lexists(model_cache_storage.trash_dir(caches.hub.resolve()))
        return
    outcome = delete.delete(ILLUSTRIOUS, planned.pin)
    assert outcome.changed
    [left] = outcome.remaining
    assert left["kind"] == "hub" and "open" in left["reason"]
    assert 0 < left["bytes"] <= next(part.bytes for part in planned.parts if part.repo == SDXL)
    # Its own repository and converted copy went; the ref went first.
    assert not os.path.lexists(caches.folder(ILLUSTRIOUS))
    assert not any(path.is_dir() for path in caches.converted.iterdir())
    assert not (caches.folder(SDXL) / "refs" / "main").exists()
    # Files missing from a pinned revision, never a ref to a missing
    # snapshot; the sibling revision and its ref byte for byte.
    readable(caches)
    assert under(caches.state(), f"home/hub/{stem(SDXL)}/snapshots/{OLD}",
                 f"home/hub/{stem(SDXL)}/refs/old") == sibling
    monkeypatch.setattr(os, "unlink", unlink)
    if where.endswith("blob"):
        # Every entry went and the emptied revision with them: the blob
        # stays unnamed, and the answer says so.
        assert "blobs folder" in left["reason"] and calls[-1].exists()
        assert not (caches.folder(SDXL) / "snapshots" / commit).exists()
        return
    # Let go, the next delete takes what stayed.
    assert "a later delete" in left["reason"]
    assert run(ILLUSTRIOUS).remaining == []
    assert not (caches.folder(SDXL) / "snapshots" / commit).exists()
    readable(caches)
    assert under(caches.state(), f"home/hub/{stem(SDXL)}/snapshots/{OLD}",
                 f"home/hub/{stem(SDXL)}/refs/old") == sibling


def test_a_stop_in_one_repository_leaves_the_next_whole(caches, monkeypatch):
    # Anima Aesthetic alone is the last holder: its own file inside the
    # repository Turbo's recipe names, and the base. A stop in the first
    # names the base as staying, byte for byte.
    caches.plant(AESTHETIC)
    caches.ref(TURBO, commit_of(caches, AESTHETIC, TURBO))
    planned = delete.plan(AESTHETIC)
    assert [part.repo for part in planned.parts] == [TURBO, ANIMA_BASE]
    base = under(caches.state(), f"home/hub/{stem(ANIMA_BASE)}")
    unlink, calls = os.unlink, []

    def refuse_the_entry(path, *args, **kwargs):
        calls.append(Path(path))
        if len(calls) == 2:
            raise in_use(path)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", refuse_the_entry)
    outcome = delete.delete(AESTHETIC, planned.pin)
    assert calls[0].parent.name == "refs" and len(calls) == 2
    first, base_part = planned.parts
    assert outcome.remaining == [
        {"kind": "hub", "bytes": first.bytes, "reason": "A file in it is open, perhaps in another program, so some "
                                                         "of its files stay; a later delete can take them."},
        {"kind": "hub", "bytes": base_part.bytes, "reason": "It stays: the delete stopped at another part."}]
    assert under(caches.state(), f"home/hub/{stem(ANIMA_BASE)}") == base
    readable(caches)
    monkeypatch.setattr(os, "unlink", unlink)
    assert run(AESTHETIC).remaining == []
    assert not os.path.lexists(caches.folder(TURBO)) and not os.path.lexists(caches.folder(ANIMA_BASE))
    readable(caches)
