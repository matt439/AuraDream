"""Source and result evidence for the isolated Kontext browser trial, without weights."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from contextlib import closing
from dataclasses import replace

import pytest
from PIL import Image, PngImagePlugin

from backend import storage
from backend.grid import Cell
from backend.instruction_edit import KONTEXT_V1, geometry
from backend.job_state import Job
from backend.schemas import GenerateRequest
from tests.instruction_edit_helpers import stored_edit
from tools.kontext_trial_fixtures import plant_fixtures, verify_result


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    data = tmp_path_factory.mktemp("kontext-trial") / "gallery"
    return data, plant_fixtures(data)


def test_original_and_all_six_resizes_match_independent_rgb_pins(planted):
    data, fixtures = planted
    pins = {
        "1:1": "28b506e70b5876cbafe51fff8b0222272adf4a395f156647ad45121377d3f42a",
        "16:9": "7e4be89fcd90b9a627cf4afa0617228498c87500cb77b796c02d23e66e36d00f",
        "9:16": "cd314b1d9c212b0d56a0bb8f166517a96f9bcbef70a0bf19d325871f56574911",
        "4:3": "c0071174302005ea5300e456d1024d47609f253f71a224dbf3d1c7b566b99dbf",
        "3:4": "88400bbb17e9d80216162ad5bfed248bd9d8681954d8002ebdcdcfb4bc36a580",
        "3:2": "bb076c89baed9a4e447b5b8e342a38ab870c33a19dbe72b1fc60486d8012f1ad",
        "2:3": "6ea22bea1b912485a40d6337816cc0660f2b402714f52fe0c5246ede09ea3baf",
    }
    assert list(fixtures) == list(pins)
    with closing(sqlite3.connect(data / "auradream.db")) as conn:
        rows = conn.execute("SELECT id, filename, width, height FROM images ORDER BY id").fetchall()
    assert len(rows) == len(pins)
    for item, (aspect, expected) in zip(rows, pins.items(), strict=True):
        image_id, filename, width, height = item
        source = fixtures[aspect]
        assert (source["id"], source["filename"], source["width"], source["height"]) == item
        with Image.open(data / "outputs" / filename) as image:
            assert image.mode == "RGB" and image.size == (width, height)
            assert hashlib.sha256(image.tobytes()).hexdigest() == source["hash"] == expected


def test_refuses_existing_gallery_and_different_source(tmp_path):
    existing = tmp_path / "gallery"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_bytes(b"existing gallery")
    with pytest.raises(FileExistsError):
        plant_fixtures(existing)
    assert list(existing.iterdir()) == [marker] and marker.read_bytes() == b"existing gallery"

    source = tmp_path / "wrong.png"
    with Image.new("RGB", (1024, 1024), "orange") as image:
        image.save(source)
    original = source.read_bytes()
    with pytest.raises(subprocess.CalledProcessError):
        plant_fixtures(tmp_path / "new", source)
    assert source.read_bytes() == original
    assert not (tmp_path / "new" / "kontext-fixtures.json").exists()


@pytest.fixture
def result(planted, tmp_path, monkeypatch, request):
    original, fixtures = planted
    data = tmp_path / "gallery"
    shutil.copytree(original, data)
    settings = replace(storage.settings, db_path=data / "auradream.db", outputs_dir=data / "outputs")
    monkeypatch.setattr(storage, "settings", settings)

    aspect, recipe, peak = getattr(request, "param", ("16:9", "kontext-v1", (12345, 23456)))
    source = fixtures[aspect]
    seed = 12345
    edit = stored_edit(storage.get_image(source["id"]), recipe=recipe)
    size = geometry(source["width"], source["height"], recipe=recipe).output_size
    edit_request = GenerateRequest(prompt="Change the red ball to a blue ball.", seed=seed, steps=28,
                                   width=size[0], height=size[1], guidance_scale=2.5, mu=None, edit=edit)
    job = Job(id="trial-job", run_id="trial-run", cells=[Cell(edit_request, seed)],
              request=edit_request, seed=seed, status="done", step=28, total_steps=28)
    with Image.new("RGB", size, (21, 35, 97)) as image:
        record = storage.save_image(image, edit_request, seed=seed, job_id=job.id, run_id=job.run_id,
                                    model_id=KONTEXT_V1.model_id, quantization="fp8", batch=1,
                                    duration_ms=17, peak=peak)
    job.images.append(record.model_dump(mode="json"))
    return (data, source, edit.recipe, seed, job.snapshot(), record.model_dump(mode="json"))


def test_verifies_terminal_job_gallery_sql_and_png(result):
    with Image.new("RGB", (512, 288), (21, 35, 97)) as expected:
        digest = hashlib.sha256(expected.tobytes()).hexdigest()
    verified = verify_result(*result, real=True)
    assert verified["rgb_sha256"] == digest
    assert (verified["vram_peak_allocated"], verified["vram_peak_reserved"]) == (12345, 23456)


@pytest.mark.parametrize("result", [("1:1", "kontext-1024-v1", None)], indirect=True)
def test_cpu_rehearsal_accepts_absent_peak_but_card_evidence_requires_one(result):
    verified = verify_result(*result, real=False)
    assert verified["dimensions"] == [1024, 1024] and verified["vram_peak_allocated"] is None
    with pytest.raises(ValueError, match="allocator"):
        verify_result(*result, real=True)


@pytest.mark.parametrize("result", [("1:1", "kontext-v1", (12345, 23456))], indirect=True)
def test_first_card_control_must_reproduce_the_historical_pixels(result):
    assert verify_result(*result, real=False)["dimensions"] == [512, 512]
    with pytest.raises(ValueError, match="control"):
        verify_result(*result, real=True)


@pytest.mark.parametrize("surface", ["job", "gallery", "sql", "png", "pixels"])
def test_a_single_disagreeing_surface_refuses_result(result, surface):
    data, _, _, _, job, record = result
    if surface == "job":
        job["images"][0]["edit"]["edit_source"]["hash"] = "0" * 64
    elif surface == "gallery":
        record["vram_peak_reserved"] += 1
    elif surface == "sql":
        with closing(sqlite3.connect(data / "auradream.db")) as conn:
            conn.execute("UPDATE images SET seed = seed + 1 WHERE id = ?", (record["id"],))
            conn.commit()
    else:
        path = data / "outputs" / record["filename"]
        with Image.open(path) as image:
            pixels = image.copy()
            metadata = json.loads(image.info["auradream"])
        if surface == "png":
            metadata["edit"]["recipe"] = "kontext-1024-v1"
        else:
            pixels.putpixel((0, 0), (255, 255, 255))
        chunks = PngImagePlugin.PngInfo()
        chunks.add_text("auradream", json.dumps(metadata))
        with pixels:
            pixels.save(path, pnginfo=chunks)
    with pytest.raises(ValueError):
        verify_result(*result, real=True)
