"""Saved drafts retain ownership and survive failed or concurrent edits."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from backend import gallery_lock, presets
from backend.schemas_presets import (
    MAX_PRESETS, PresetFields, PresetGrid, PresetPayload, PresetUpdate,
)
from backend.seeds import MAX_BATCH, MAX_GRID_SEEDS, MAX_RUNS, MAX_SEED
from tests.bundle_helpers import holding_child


def fields(**payload):
    return PresetFields(
        name="  夜の setup  ", model_id="black-forest-labs/FLUX.1-Krea-dev",
        payload=PresetPayload(prompt="  a cat\n猫  ", negative_prompt="  bad\n悪い  ", **payload),
    )


def test_first_read_does_not_create_a_directory(tmp_path):
    database = tmp_path / "not-created" / "gallery.db"
    assert presets.read(database) == []
    assert not database.parent.exists()


def test_new_feature_uses_only_auradream_name(tmp_path):
    database = tmp_path / "local-diffusion.db"
    legacy = tmp_path / "local-diffusion.presets.json"
    legacy.write_text("not a saved setup", encoding="utf-8")
    assert presets.path_for(database) == tmp_path / "auradream.presets.json"
    assert presets.read(database) == []
    presets.change(database, fields())
    assert legacy.read_text(encoding="utf-8") == "not a saved setup"


def test_exact_draft_and_sampling_ownership_round_trip(tmp_path):
    database = tmp_path / "gallery.db"
    original = fields(guidance_scale=0, mu=0, seed=None, num_images=2, runs=3)
    saved = presets.change(database, original)[0]
    assert saved.name == "夜の setup"
    assert saved.payload == original.payload
    assert saved.payload.prompt == "  a cat\n猫  "
    assert saved.payload.negative_prompt == "  bad\n悪い  "
    assert saved.payload.steps is None and saved.payload.true_cfg is None
    assert saved.payload.seed is None and saved.payload.guidance_scale == 0
    assert saved.payload.mu == 0 and saved.revision == 1
    assert presets.read(database) == [saved]
    assert not database.exists()
    assert not (tmp_path / "auradream.settings.json").exists()


def test_grid_text_order_and_inactive_repeat_round_trip(tmp_path):
    database = tmp_path / "gallery.db"
    grid = PresetGrid(
        prompt_axis="  light = night | day\ncolour = 青 | red  ", prompt_mode="random",
        steps_axis="28,12", cfg_axis="3,0", mu_axis="1.2,0", size_axis="1024x768,768x1024",
        seeds=5, seed_axis="44,11,22", budget_minutes=45,
    )
    saved = presets.change(database, fields(mode="grid", grid=grid, runs=7))[0]
    assert saved.payload.grid == grid and saved.payload.runs == 7
    assert presets.read(database)[0] == saved


def test_revisions_update_delete_and_missing_item(tmp_path):
    database = tmp_path / "gallery.db"
    original = presets.change(database, fields())[0]
    update = PresetUpdate(**fields(seed=MAX_SEED).model_dump(), revision=1)
    updated = presets.change(database, update, item_id=original.id)[0]
    assert updated.id == original.id and updated.revision == 2
    assert updated.payload.seed == MAX_SEED
    for values in (update, None):
        with pytest.raises(presets.PresetsError, match="changed in another tab") as error:
            presets.change(database, values, item_id=original.id, revision=1)
        assert error.value.status == 409
        assert presets.read(database) == [updated]
    assert presets.change(database, None, item_id=original.id, revision=2) == []
    with pytest.raises(presets.PresetsError, match="no longer") as error:
        presets.change(database, None, item_id=original.id, revision=2)
    assert error.value.status == 404


@pytest.mark.parametrize("payload", [
    {"seed": -1}, {"seed": MAX_SEED + 1}, {"seed": True}, {"seed": "1"},
    {"num_images": 0}, {"num_images": MAX_BATCH + 1}, {"runs": MAX_RUNS + 1},
    {"steps": 0}, {"steps": 101}, {"steps": 1.0}, {"steps": False},
    {"guidance_scale": -0.1}, {"guidance_scale": 20.1}, {"guidance_scale": float("nan")},
    {"mu": 10.1}, {"mu": float("inf")}, {"true_cfg": 0},
    {"width": 255}, {"height": True}, {"prompt": ""}, {"prompt": "x" * 4001},
    {"negative_prompt": "x" * 4001}, {"prompt": "\ud800"},
    {"execution": None}, {"step_cache": None}, {"reference": []}, {"lora": None},
    {"refine": None}, {"then_refine": None}, {"edit": None}, {"resume": False},
    {"mode": "grid"}, {"grid": {}},
])
def test_payload_is_typed_bounded_and_has_no_unowned_fields(payload):
    with pytest.raises(ValidationError):
        PresetPayload.model_validate({"prompt": "a cat", **payload})


@pytest.mark.parametrize("payload", [
    {"seeds": 0}, {"seeds": MAX_GRID_SEEDS + 1}, {"seeds": True},
    {"prompt_axis": "x" * 2001}, {"size_axis": "x" * 2001}, {"seed_axis": "x" * 2001},
    {"steps_axis": "x" * 201}, {"cfg_axis": "x" * 201}, {"mu_axis": "x" * 201},
    {"prompt_mode": "other"}, {"budget_minutes": 0}, {"budget_minutes": 10081},
    {"budget_minutes": float("inf")}, {"resume": False}, {"lora_scale_axis": ""},
    {"prompt_axis": "\ud800"},
])
def test_grid_field_limits_and_unknown_fields(payload):
    with pytest.raises(ValidationError):
        PresetGrid.model_validate(payload)


@pytest.mark.parametrize("changes", [
    {"name": " "}, {"name": "x" * 81}, {"name": "my\nsetup"}, {"name": "\ud800"},
    {"model_id": " "}, {"model_id": " model"}, {"model_id": "model\t"}, {"model_id": "\ud800"},
    {"model_id": "x" * 513}, {"id": "a" * 32}, {"revision": 1}, {"execution": None},
])
def test_item_fields_do_not_accept_ids_revisions_or_unreadable_labels(changes):
    with pytest.raises(ValidationError):
        PresetFields.model_validate({**fields().model_dump(), **changes})


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_atomic_failure_keeps_original_and_removes_temporary(tmp_path, monkeypatch, failure):
    database = tmp_path / "gallery.db"
    presets.change(database, fields())
    path = presets.path_for(database)
    before = path.read_bytes()

    def fail(*args):
        raise PermissionError("test failure")

    monkeypatch.setattr(presets.os, failure, fail)
    with pytest.raises(presets.PresetsError, match="previous saved list"):
        presets.change(database, fields(seed=123))
    assert path.read_bytes() == before
    assert list(tmp_path.glob(".auradream.presets.*.tmp")) == []


@pytest.mark.parametrize("data", [
    b"not JSON", b'{"version":2,"items":[]}', b'{"version":true,"items":[]}',
    b'{"version":1,"items":[],"unknown":true}', b'{"version":1,"items":{},"unknown":true}',
    b'{"version":1,"version":1,"items":[]}', b'{"version":1,"items":[{}]}',
    b"[" * 2000 + b"]" * 2000,
])
def test_bad_file_is_reported_and_preserved_on_edit(tmp_path, data):
    database = tmp_path / "gallery.db"
    path = presets.path_for(database)
    path.write_bytes(data)
    for call in (lambda: presets.read(database), lambda: presets.change(database, fields())):
        with pytest.raises(presets.PresetsError, match="left unchanged"):
            call()
        assert path.read_bytes() == data


def test_unknown_saved_payload_field_and_duplicate_ids_preserve_file(tmp_path):
    database = tmp_path / "gallery.db"
    original = presets.change(database, fields())[0].model_dump()
    path = presets.path_for(database)
    for items in ([original, original], [{**original, "payload": {**original["payload"], "execution": None}}]):
        data = json.dumps({"version": 1, "items": items}).encode("utf-8")
        path.write_bytes(data)
        with pytest.raises(presets.PresetsError, match="left unchanged"):
            presets.change(database, fields())
        assert path.read_bytes() == data


def test_file_byte_bounds_fail_before_replacing_previous_file(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    presets.change(database, fields())
    path = presets.path_for(database)
    before = path.read_bytes()
    monkeypatch.setattr(presets, "MAX_FILE_BYTES", len(before) + 1)
    with pytest.raises(presets.PresetsError, match="exceed 8 MiB") as error:
        presets.change(database, fields(seed=2))
    assert error.value.status == 422 and path.read_bytes() == before
    monkeypatch.setattr(presets, "MAX_FILE_BYTES", len(before) - 1)
    with pytest.raises(presets.PresetsError, match="left unchanged"):
        presets.read(database)


def test_collection_limit_on_read_and_create(tmp_path):
    database = tmp_path / "gallery.db"
    original = presets.change(database, fields())[0]
    full = [original.model_copy(update={"id": f"{number:032x}"}) for number in range(MAX_PRESETS)]
    path = presets.path_for(database)
    presets._write(path, full)
    before = path.read_bytes()
    with pytest.raises(presets.PresetsError, match="at most") as error:
        presets.change(database, fields())
    assert error.value.status == 422 and path.read_bytes() == before
    presets._write(path, [*full, original.model_copy(update={"id": "f" * 32})])
    with pytest.raises(presets.PresetsError, match="left unchanged"):
        presets.read(database)


def test_concurrent_edits_accept_one_revision(tmp_path):
    database = tmp_path / "gallery.db"
    original = presets.change(database, fields())[0]
    start = Barrier(2)

    def update(seed):
        start.wait(timeout=15)
        try:
            return presets.change(database, PresetUpdate(
                **fields(seed=seed).model_dump(), revision=1,
            ), item_id=original.id)
        except presets.PresetsError as exc:
            return exc.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(update, [1, 2]))
    saved = [outcome for outcome in outcomes if isinstance(outcome, list)]
    assert len(saved) == 1 and outcomes.count(409) == 1
    assert presets.read(database) == saved[0]
    assert saved[0][0].revision == 2


def test_another_process_sees_setup_lock_but_not_gallery_lock(tmp_path, monkeypatch):
    database = tmp_path / "gallery.db"
    # A held setup lock refuses only after the writer's wait, read per call.
    monkeypatch.setattr(gallery_lock, "SESSION_WAIT", 0.3)
    child = holding_child(presets.path_for(database))
    try:
        with pytest.raises(presets.PresetsError, match="Another AuraDream process") as error:
            presets.change(database, fields())
        assert error.value.status == 409
        assert presets.read(database) == []
        with gallery_lock.hold(database):
            pass
    finally:
        child.stdin.close()
        child.wait(timeout=15)
    with gallery_lock.hold(database):
        assert len(presets.change(database, fields())) == 1


def test_concurrent_process_creates_keep_both_items(tmp_path):
    database = tmp_path / "gallery.db"
    code = """
from pathlib import Path
import sys
from backend.presets import change
from backend.schemas_presets import PresetFields, PresetPayload
print('ready', flush=True)
sys.stdin.readline()
change(Path(sys.argv[1]), PresetFields(name=sys.argv[2], model_id='model', payload=PresetPayload(prompt=sys.argv[2])))
"""
    children = [subprocess.Popen(
        [sys.executable, "-c", code, str(database), name],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for name in ("first", "second")]
    try:
        for child in children:
            assert child.stdout.readline().strip() == "ready"
        for child in children:
            child.stdin.write("\n")
            child.stdin.flush()
        for child in children:
            stdout, stderr = child.communicate(timeout=30)
            assert child.returncode == 0, stdout + stderr
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
    assert {item.name for item in presets.read(database)} == {"first", "second"}
