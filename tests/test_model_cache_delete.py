"""The checkpoint delete's planner (`GEN#72` stage 1): what deleting one
catalogue entry would take, read-only, on scratch caches in both layouts.

docs/maintainer/plan-model-delete.md is the spec; its Tests section lists
these cases. Every test runs on a plain-copy cache (Windows) and, where the
platform lets the test make links, a symlink one (`blobs/` behind
`snapshots/`). The recipes are the real ones.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from backend import model_cache_delete as delete, model_cache_storage, model_info
from backend.backends import anima_loading, single_file
from backend.encoder_table import TokenizerSource
from backend.launch import CHECKPOINTS
from tests.model_cache_fixtures import LAYOUTS, link_folder, scratch_caches

TURBO = "circlestone-labs/Anima"
AESTHETIC = "circlestone-labs/Anima@aesthetic-v1.1"
PREVIEW = "Gazingstars123/Anima-2.9B"
ANIMA_BASE = "circlestone-labs/Anima-Base-v1.0-Diffusers"
ILLUSTRIOUS, WAI, ROUWEI = single_file.ILLUSTRIOUS_ID, single_file.WAI_ID, single_file.ROUWEI_ID
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
ERNIE = "baidu/ERNIE-Image-Turbo"
KREA2 = "krea/Krea-2-Turbo"
ZANIME, ZIMAGE = "SeeSee21/Z-Anime", "Tongyi-MAI/Z-Image"
NETAYUME, LUMINA = "duongve/NetaYume-Lumina-Image-2.0", "Alpha-VLLM/Lumina-Image-2.0"
OLD = "8" * 40


@pytest.fixture(params=LAYOUTS)
def caches(request, tmp_path, monkeypatch):
    return scratch_caches(tmp_path, monkeypatch, request.param)


def plan(model: str, **context) -> delete.Plan:
    return delete.plan(model, delete.PlanContext(**context))


def written(repo: str, names) -> int:
    """The bytes `Caches.put` writes for `names` of `repo` by default."""
    return sum(len(f"{repo}:{name}".encode()) for name in names)


def selected_names(caches, model: str, repo: str) -> list[str]:
    return next(names for source, _, names in caches.recipe(model) if source == repo)


def relatives(result: delete.Plan) -> set[str]:
    return {removal.relative for removal in result.removals}


# ------------------------------------------------------------- Anima's three


def test_anima_aesthetic_keeps_the_base_its_sharers_use_and_turbos_revision(caches):
    for model in (TURBO, AESTHETIC, PREVIEW):
        caches.plant(model)
        caches.convert(model)
    before = caches.state()
    result = plan(AESTHETIC)
    repo, commit, name = caches.own_file(AESTHETIC)
    _, turbo_commit, _ = caches.own_file(TURBO)
    assert {(part.kind, part.repo, part.revision, part.name) for part in result.parts} == {
        ("hub", TURBO, commit, None),
        ("converted", None, None, anima_loading._target(AESTHETIC)[0].name),
    }
    hub = next(part for part in result.parts if part.kind == "hub")
    assert hub.shared_repository and not hub.whole and hub.files == 1
    assert hub.bytes == written(repo, [name])
    assert result.kept_shared == [{
        "bytes": written(ANIMA_BASE, selected_names(caches, AESTHETIC, ANIMA_BASE)),
        "with": ["Anima 2.9B Preview v1", "Anima Turbo v1.1"],
    }]
    assert result.also_taken_shared == []
    # Its file sits in Turbo's repository: file-level deletion is stage 4.
    assert result.held is not None and result.held["code"] == "model_delete_held"
    assert "Anima Turbo v1.1" in result.held["detail"] and "not built yet" in result.held["detail"]
    # Nothing of Turbo's revision, the base or another converted tree.
    assert not any(turbo_commit in path or "Anima-Base" in path or "turbo" in path for path in relatives(result))
    assert caches.state() == before


def test_anima_preview_owns_its_repository_and_keeps_the_base(caches):
    for model in (TURBO, AESTHETIC, PREVIEW):
        caches.plant(model)
    caches.convert(PREVIEW)
    result = plan(PREVIEW)
    assert {(part.kind, part.repo, part.whole) for part in result.parts} == {
        ("hub", PREVIEW, True), ("converted", None, False)}
    assert result.kept_shared[0]["with"] == ["Anima Aesthetic v1.1", "Anima Turbo v1.1"]
    assert result.held is None
    assert result.frees_bytes == sum(part.bytes for part in result.parts) > 0


def test_a_blob_another_revision_still_resolves_to_stays(caches):
    # On a symlink cache one blob can serve two revisions: Aesthetic's file
    # with Turbo's bytes. Only the link goes; a plain copy frees its bytes.
    turbo_repo, turbo_commit, turbo_name = caches.own_file(TURBO)
    repo, commit, name = caches.own_file(AESTHETIC)
    same = b"the same bytes at two commits"
    caches.put(turbo_repo, turbo_commit, turbo_name, same)
    caches.put(repo, commit, name, same)
    result = plan(AESTHETIC)
    [part] = result.parts
    if caches.layout == "symlink":
        assert part.bytes == 0 and not any(path.split("/")[1] == "blobs" for path in relatives(result))
    else:
        assert part.bytes == len(same)
    assert relatives(result) >= {f"models--circlestone-labs--Anima/snapshots/{commit}/{name}"}


def test_a_stage_4_switch_releases_the_shared_repository_hold(caches, monkeypatch):
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    assert plan(AESTHETIC).held is not None
    monkeypatch.setattr(delete, "SHARED_REPOSITORY_DELETE", True)
    assert plan(AESTHETIC).held is None


# --------------------------------------------- the SDXL three, last holder


def _illustrious_alone(caches) -> int:
    caches.plant(ILLUSTRIOUS)
    return written(SDXL, selected_names(caches, ILLUSTRIOUS, SDXL))


def test_the_last_holder_takes_the_shared_base_and_says_so(caches):
    base = _illustrious_alone(caches)
    result = plan(ILLUSTRIOUS)
    assert result.kept_shared == []
    assert result.also_taken_shared == [{"bytes": base, "with": ["RouWei 0.8 epsilon", "WAI Illustrious v17"]}]
    sdxl = next(part for part in result.parts if part.repo == SDXL)
    assert sdxl.shared_repository and sdxl.bytes == base and sdxl.files == 17
    own = next(part for part in result.parts if part.repo == ILLUSTRIOUS)
    assert own.whole and not own.shared_repository
    # Taking files out of the SDXL repository others name waits for stage 4.
    assert result.held is not None and "WAI Illustrious v17" in result.held["detail"]


@pytest.mark.parametrize("present", ["own file", "other revision", "converted", "selected", "loaded",
                                     "saved for the next start", "a tokenizer fetch"])
def test_a_present_sharer_keeps_the_shared_base(caches, present):
    base = _illustrious_alone(caches)
    context = {}
    if present == "own file":
        caches.plant(WAI, only=WAI)
    elif present == "other revision":
        repo, _, name = caches.own_file(WAI)
        caches.put(repo, OLD, name)  # GEN#71: cached only at a commit no record names
    elif present == "converted":
        caches.convert(WAI)
    elif present == "selected":
        context["selected"] = WAI
    elif present == "loaded":
        context["loaded"] = WAI
    elif present == "saved for the next start":
        context["saved"] = WAI
    else:
        _, commit, _ = next(source for source in caches.recipe(WAI) if source[0] == SDXL)
        context["pending"] = frozenset({TokenizerSource(SDXL, "tokenizer", revision=commit, tree=WAI)})
    result = plan(ILLUSTRIOUS, **context)
    assert result.also_taken_shared == []
    expected = ["RouWei 0.8 epsilon", "WAI Illustrious v17"] if present == "a tokenizer fetch" else ["WAI Illustrious v17"]
    assert result.kept_shared == [{"bytes": base, "with": expected}]
    assert not any("stable-diffusion-xl" in path for path in relatives(result))
    assert {part.repo for part in result.parts} == {ILLUSTRIOUS}
    # Nothing comes out of the shared repository, so nothing waits for stage 4.
    assert result.held is None
    if present == "a tokenizer fetch":
        assert result.fetching(context["pending"])


# ------------------------------------------------ a repository owned whole


def test_an_owned_repository_goes_whole_every_revision_and_leftover(caches):
    _, commit, _ = caches.recipe(ERNIE)[0]
    caches.plant(ERNIE)
    caches.ref(ERNIE, commit)
    for name in ("README.md", "pe/model.safetensors", "pe_tokenizer/tokenizer.json"):
        caches.put(ERNIE, commit, name)  # unselected: the prompt rewriter, the card
    caches.put(ERNIE, OLD, "transformer/config.json", b"an older download")
    missing = caches.folder(ERNIE) / ".no_exist" / commit / "model.fp16.safetensors"
    missing.parent.mkdir(parents=True)
    missing.write_bytes(b"")
    result = plan(ERNIE)
    assert len(result.parts) == 1
    part = result.parts[0]
    assert (part.kind, part.repo, part.revision, part.whole, part.shared_repository) == ("hub", ERNIE, None, True, False)
    folder = caches.folder(ERNIE)
    on_disk = [path for path in folder.rglob("*") if path.is_file() and not path.is_symlink()]
    assert part.bytes == sum(path.stat().st_size for path in on_disk)
    assert part.files == 12 + 3 + 1  # every snapshot entry, both revisions
    stem = "models--baidu--ERNIE-Image-Turbo"
    assert {f"{stem}/snapshots/{OLD}/transformer/config.json", f"{stem}/.no_exist/{commit}/model.fp16.safetensors",
            f"{stem}/refs/main", f"{stem}/snapshots/{commit}/pe/model.safetensors"} <= relatives(result)
    assert part.units == [folder]
    assert result.held is None and result.downloading == [] and result.kept_shared == []


def test_a_checkpoint_cached_only_at_an_unrecorded_commit_is_its_own(caches):
    caches.put(KREA2, OLD, "model_index.json")  # GEN#71's other revision
    result = plan(KREA2)
    assert [(part.kind, part.repo, part.whole) for part in result.parts] == [("hub", KREA2, True)]
    assert result.held is None and result.frees_bytes == written(KREA2, ["model_index.json"])


def test_an_unrecorded_revision_several_entries_could_own_is_held(caches):
    caches.put(TURBO, OLD, "split_files/diffusion_models/anima-turbo-v1.0.safetensors")
    for model, other in ((AESTHETIC, "Anima Turbo v1.1"), (TURBO, "Anima Aesthetic v1.1")):
        result = plan(model)
        assert result.held is not None and result.held["code"] == "model_delete_held"
        assert "could also own" in result.held["detail"] and other in result.held["detail"]
        assert result.removals == []


def test_lite_repositories_and_an_alias_go_with_their_entry(caches):
    entry = model_info.catalogue_entry(KREA2)
    assert entry is not None
    nvfp4, int4, alias = entry.lite_repo("nvfp4"), entry.lite_repo("int4"), entry.lite_alias("int4")
    assert nvfp4 and int4 and alias and alias != int4
    caches.plant(KREA2)
    caches.plant(nvfp4)
    caches.put(alias, "6" * 40, "tokenizer/tokenizer_config.json")
    result = plan(KREA2)
    assert {(part.kind, part.repo, part.whole) for part in result.parts} == {
        ("hub", KREA2, True), ("lite", nvfp4, True), ("lite", alias, True)}
    assert result.model_ids == tuple(sorted({KREA2, nvfp4, int4}))
    assert {KREA2, nvfp4, int4, alias} <= result.repositories
    assert result.held is None
    # A lite id is its entry's sub-line, never a unit of its own.
    with pytest.raises(delete.NotOnDisk):
        plan(nvfp4)


def test_a_shared_xet_store_is_held(caches):
    caches.plant(KREA2)
    store = caches.hub / "blobs"
    store.mkdir()
    (store / delete.SHARED_STORE_MARKER).write_text("1\n")
    result = plan(KREA2)
    assert result.held is not None and "shared Xet store" in result.held["detail"]


def test_a_shared_xet_store_holds_only_a_plan_that_takes_from_the_hub(caches):
    store = caches.hub / "blobs"
    store.mkdir()
    (store / delete.SHARED_STORE_MARKER).write_text("1\n")
    # Nothing cached is still the one 404, marker or not.
    with pytest.raises(delete.NotOnDisk):
        plan(KREA2)
    # A converted copy alone is not the store's.
    caches.convert(AESTHETIC)
    result = plan(AESTHETIC)
    assert [part.kind for part in result.parts] == ["converted"] and result.held is None


def test_a_partial_download_is_named_relative_to_the_cache(caches):
    caches.plant(KREA2)
    partial = caches.folder(KREA2) / "blobs" / "0123abcd.incomplete"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    result = plan(KREA2)
    assert result.downloading == [{"repo": KREA2, "file": "models--krea--Krea-2-Turbo/blobs/0123abcd.incomplete"}]


# ----------------------------------------------------------- converted trees


def test_the_converted_copy_takes_its_work_directories_and_nothing_alike(caches):
    caches.plant(WAI, only=WAI)
    target = caches.convert(WAI)
    work = caches.converted / f"{target.name}-abcd1234"
    (work / "tree").mkdir(parents=True)
    (work / "tree" / "partial.safetensors").write_bytes(b"half a conversion")
    lookalike = caches.converted / f"{target.name}-zzzz9999"
    lookalike.mkdir()
    (lookalike / "notes.txt").write_text("a user's own folder")
    result = plan(WAI)
    part = next(part for part in result.parts if part.kind == "converted")
    assert part.name == target.name and part.units == [target, work]
    assert part.files == 4  # the tree's three and the work directory's one
    assert not any("zzzz9999" in path for path in relatives(result))
    # Locks stay (How files are removed): the zero-byte `.lock` is no part.
    (caches.converted / f"{target.name}.lock").write_bytes(b"")
    assert not any(path.endswith(".lock") for path in relatives(plan(WAI)))


def test_no_converted_cache_yet_holds_nothing(caches):
    caches.converted.rmdir()
    caches.plant(WAI, only=WAI)
    result = plan(WAI)
    assert result.held is None and [part.kind for part in result.parts] == ["hub"]
    assert not caches.converted.exists()


def test_older_recipes_are_the_entrys_only_by_their_markers_source(caches):
    caches.plant(WAI, only=WAI)
    mine = caches.older("wai-illustrious-v17-0000000000000000",
                        json.dumps({"version": 0, "source": single_file.WAI_SOURCE}))
    caches.older("wai-illustrious-v17-1111111111111111", None)  # no marker: not referenced
    caches.older("wai-illustrious-v17-2222222222222222", "not json")
    caches.older("rouwei-old", json.dumps({"version": 0, "source": single_file.ROUWEI_SOURCE}))
    # The shared config repository is never the source.
    caches.older("sdxl-config-only", json.dumps({"version": 0, "source": single_file.SDXL_CONFIG}))
    result = plan(WAI)
    assert [(part.kind, part.name) for part in result.parts if part.kind == "converted-old"] == [
        ("converted-old", mine.name)]
    assert result.held is None


def test_an_older_turbo_tree_stays_beside_an_aesthetic_delete(caches):
    caches.convert(AESTHETIC)
    older = caches.older("anima-turbo-v1.1-0000000000000000",
                         json.dumps({"version": 0, "source": anima_loading.SOURCE}))
    aesthetic = plan(AESTHETIC)
    assert not any(older.name in path for path in relatives(aesthetic))
    assert aesthetic.held is None
    turbo = plan(TURBO)
    assert [(part.kind, part.name) for part in turbo.parts] == [("converted-old", older.name)]


def test_an_older_tree_from_a_shared_repository_and_neither_file_is_held(caches):
    caches.convert(AESTHETIC)
    caches.older("anima-preview3-0000000000000000", json.dumps({"version": 0, "source": {
        "repo_id": TURBO, "filename": "split_files/diffusion_models/anima-preview3.safetensors"}}))
    for model in (AESTHETIC, TURBO):
        result = plan(model)
        assert result.held is not None and "neither uses now" in result.held["detail"]
        assert not any("preview3" in path for path in relatives(result))


# ------------------------------------------------------ unknown and absent


def test_an_unknown_id_and_an_absent_entry_are_one_answer(caches):
    for model in ("nobody/nothing", KREA2, ILLUSTRIOUS):
        with pytest.raises(delete.NotOnDisk):
            plan(model)


# ------------------------------------- the selected checkpoint outside it


@pytest.mark.parametrize("entry, other", [(ZANIME, ZIMAGE), (NETAYUME, LUMINA)])
def test_an_id_outside_the_catalogue_is_the_repository_it_names(caches, entry, other):
    # Any text-to-image checkpoint runs by id; Z-Image and Lumina are each
    # one entry's recipe repository, so the plan takes them whole.
    caches.plant(entry)
    result = plan(entry)
    assert (("hub", other, True) in {(part.kind, part.repo, part.whole) for part in result.parts})
    for state, key in (("selected", "selected"), ("loaded", "loaded"), ("saved", "load_on_startup")):
        assert plan(entry, **{state: other}).answer()[key] is True, state
    # The Hub's ids, and NTFS's folders, ignore case.
    assert plan(entry, selected=other.upper()).answer()["selected"] is True
    assert plan(entry, selected=f"{other}@main").answer()["selected"] is True
    assert plan(entry, selected="someone/else").answer()["selected"] is False


def test_a_case_variant_of_a_catalogue_id_is_its_entry(caches):
    caches.plant(KREA2)
    assert plan(KREA2, selected=KREA2.lower(), loaded=KREA2.upper()).answer()["selected"] is True
    assert plan(KREA2, loaded=KREA2.upper()).answer()["loaded"] is True


def test_another_catalogue_entry_selected_is_a_sharer_not_the_plan(caches):
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    _illustrious_alone(caches)
    assert plan(TURBO, selected=AESTHETIC, loaded=AESTHETIC).answer()["selected"] is False
    assert plan(ILLUSTRIOUS, selected=WAI).answer()["selected"] is False


# ------------------------------------------------------------ containment


def test_paths_on_two_drives_are_not_inside(monkeypatch):
    if sys.platform == "win32":
        assert not delete._lexically_inside(Path("D:/hub/models--a--b"), Path("C:/hub"))

    def two_drives(paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(os.path, "commonpath", two_drives)
    assert not delete._lexically_inside(Path("/one/hub/models--a--b"), Path("/two/hub"))


def test_a_converted_target_outside_the_root_holds_and_never_answers_503(caches, monkeypatch, tmp_path):
    # As the fake's: this server's converted root on another drive than the
    # HF_HOME the loaders resolve their targets through.
    elsewhere = tmp_path / "other-drive"
    monkeypatch.setattr(model_cache_storage, "hf_home_dir", lambda name: elsewhere / name)
    commonpath = os.path.commonpath
    mark = os.path.normcase(str(elsewhere))

    def two_drives(paths):
        if len({os.path.normcase(os.path.abspath(path)).startswith(mark) for path in paths}) > 1:
            raise ValueError("Paths don't have the same drive")
        return commonpath(paths)

    monkeypatch.setattr(os.path, "commonpath", two_drives)
    # Nothing cached: the one 404, though the target is held unlooked at.
    with pytest.raises(delete.NotOnDisk):
        plan(WAI)
    caches.plant(WAI, only=WAI)
    result = plan(WAI)
    assert [part.kind for part in result.parts] == ["hub"]
    assert result.held is not None and "outside this server's converted cache" in result.held["detail"]


def test_a_hub_reached_through_a_link_plans_as_its_real_folder(caches, monkeypatch, tmp_path):
    # `~/.cache` or HF_HOME linked to a larger disk: the scan resolves the
    # link, and the plan's folders must be the folders it scanned.
    for model in (TURBO, AESTHETIC):
        caches.plant(model)
    real = plan(TURBO)
    assert real.held is not None and "Anima Aesthetic v1.1 also use" in real.held["detail"]
    linked = tmp_path / "linked" / "hub"
    link_folder(linked, caches.hub)
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(linked))
    monkeypatch.setenv("HF_HUB_CACHE", str(linked))
    assert model_cache_storage.hub_cache_dir() == linked
    through = plan(TURBO)
    assert through.answer() == real.answer()
    assert through.parts and relatives(through) == relatives(real)
    for removal in through.removals:
        assert removal.path.is_relative_to(caches.hub.resolve()), removal


def test_the_scratch_roots_are_not_the_real_cache_and_the_sentinel_survives(caches):
    real = caches.real_home.resolve()
    assert caches.home.resolve() != real and not caches.home.resolve().is_relative_to(real)
    assert model_cache_storage.hub_cache_dir() == caches.hub
    assert single_file.cache_root() == caches.converted
    sentinel = caches.state()
    for model in (TURBO, AESTHETIC, PREVIEW, ILLUSTRIOUS, WAI, ERNIE, KREA2):
        caches.plant(model)
    for model in (AESTHETIC, WAI):
        caches.convert(model)
    sentinel = {key: value for key, value in caches.state().items() if key.startswith("real/")}
    planned = [plan(model) for model in (TURBO, AESTHETIC, PREVIEW, ILLUSTRIOUS, WAI, ERNIE, KREA2)]
    for result in planned:
        for removal in result.removals:
            assert removal.path.resolve().is_relative_to(caches.home.resolve()), removal
            assert not removal.path.resolve().is_relative_to(caches.sentinel.resolve())
    assert {key: value for key, value in caches.state().items() if key.startswith("real/")} == sentinel


def test_a_repository_folder_linked_out_of_the_cache_is_held(caches):
    # A junction (Windows) or symlink from the Hub cache into the sentinel
    # home: the scan follows it, the plan never takes it.
    link_folder(caches.folder("krea/Krea-2-Raw"), caches.sentinel / "hub" / "models--krea--Krea-2-Turbo")
    before = caches.state()
    result = plan("krea/Krea-2-Raw")
    assert result.held is not None and "outside the Hub cache" in result.held["detail"]
    assert result.parts == [] and result.removals == []
    assert caches.state() == before


# ---------------------------------------------------------- no mutation, pin


def test_planning_every_entry_changes_nothing_byte_for_byte(caches):
    for model in (TURBO, AESTHETIC, PREVIEW, ILLUSTRIOUS, WAI, ERNIE, KREA2):
        caches.plant(model)
    for model in (TURBO, WAI):
        caches.convert(model)
    caches.older("wai-illustrious-v17-0000000000000000",
                 json.dumps({"version": 0, "source": single_file.WAI_SOURCE}))
    partial = caches.folder(KREA2) / "blobs" / "partial.incomplete"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")
    before = caches.state()
    planned = 0
    for entry in CHECKPOINTS:
        try:
            plan(entry.repo)
            planned += 1
        except delete.NotOnDisk:
            pass
        model_cache_storage.snapshot()
    assert planned >= 7
    assert caches.state() == before


def test_the_pin_is_stable_until_the_cache_changes(caches):
    for model in (ILLUSTRIOUS, ERNIE):
        caches.plant(model)
    caches.plant(WAI, only=WAI)
    first, second = plan(ERNIE), plan(ERNIE)
    assert first.pin == second.pin and len(first.pin) == 64
    assert plan(ILLUSTRIOUS).pin == plan(ILLUSTRIOUS).pin
    caches.put(ERNIE, OLD, "model_index.json", b"a file added between the plan and the delete")
    assert plan(ERNIE).pin != first.pin
    # A kept file changing changes the pin too: what was kept is part of the question.
    kept = plan(ILLUSTRIOUS)
    _, commit, names = next(source for source in caches.recipe(ILLUSTRIOUS) if source[0] == SDXL)
    entry = caches.folder(SDXL) / "snapshots" / commit / names[0]
    stamp = entry.lstat()
    # The link itself on a symlink cache: its blob may serve the delete's files too.
    os.utime(entry, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 10_000_000_000),
             **({"follow_symlinks": False} if entry.is_symlink() else {}))
    assert plan(ILLUSTRIOUS).pin != kept.pin


# ------------------------------------------------------------------ answers


def test_the_answer_names_no_absolute_path(caches, tmp_path):
    for model in (TURBO, AESTHETIC, ERNIE):
        caches.plant(model)
    caches.convert(AESTHETIC)
    for model in (AESTHETIC, ERNIE):
        text = json.dumps(plan(model).answer())
        for root in (tmp_path, caches.home):
            assert str(root) not in text and Path(root).as_posix() not in text
            assert str(root).replace("\\", "\\\\") not in text
    answer = plan(ERNIE, selected=KREA2, saved=ERNIE).answer()
    assert set(answer) == {"entry", "label", "parts", "kept_shared", "also_taken_shared", "frees_bytes",
                           "held", "downloading", "selected", "loaded", "load_on_startup", "pin"}
    assert (answer["entry"], answer["label"]) == (ERNIE, "ERNIE-Image Turbo")
    assert (answer["selected"], answer["loaded"], answer["load_on_startup"]) == (False, False, True)


def test_a_plan_waits_for_no_scan_and_releases_the_reservation(caches):
    caches.plant(ERNIE)
    model_cache_storage._scan_lock.acquire()
    try:
        with pytest.raises(model_cache_storage.ScanBusy):
            plan(ERNIE)
    finally:
        model_cache_storage._scan_lock.release()
    plan(ERNIE)
    assert not model_cache_storage._scan_lock.locked()


def test_the_index_keeps_first_hops_behind_an_unchanged_answer(caches):
    caches.plant(ERNIE)
    with model_cache_storage._scan_lock:
        answer, index = model_cache_storage._scan(caches.hub)
        again = model_cache_storage._snapshot(caches.hub)
    # The drive's free space moves under a parallel run; the rest is one answer.
    moving = {"free_bytes", "converted_free_bytes"}
    assert {key: value for key, value in answer.items() if key not in moving} == {
        key: value for key, value in again.items() if key not in moving}
    files = [file for file in index.files if file.repo_id == ERNIE]
    assert len(files) == 12 and all(file.owners == {ERNIE} for file in files)
    for file in files:
        assert file.entry.is_relative_to(caches.folder(ERNIE) / "snapshots")
        if caches.layout == "symlink":
            assert file.entry.is_symlink() and file.blob.parent == caches.folder(ERNIE) / "blobs"
        else:
            assert file.blob == file.entry
    assert index.recipe_repos[ERNIE] == {ERNIE}
    assert index.converted_root == caches.converted
