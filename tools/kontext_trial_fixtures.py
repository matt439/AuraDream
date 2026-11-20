"""Plant the owned INP#13 source and verify results against independent evidence."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/card/kontext-house-source.png"
CONTROL_RGB = "7e0062553284e700b6755b4c0450b7a8da1f43900c26ba5ed1f34f9c53512689"
INSTRUCTION = "Change the red ball to a blue ball."
MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
FIXTURE_MODEL = "auradream-owned/synthetic-inp11-v2"
MANIFEST = "kontext-fixtures.json"

# Independent pins from the frozen trial, including its November 18 control.
# Values are original width/height, kontext-v1 output size, and original RGB.
CASES = {
    "1:1": ((1024, 1024), (512, 512), "28b506e70b5876cbafe51fff8b0222272adf4a395f156647ad45121377d3f42a"),
    "16:9": ((1024, 576), (512, 288), "7e4be89fcd90b9a627cf4afa0617228498c87500cb77b796c02d23e66e36d00f"),
    "9:16": ((576, 1024), (288, 512), "cd314b1d9c212b0d56a0bb8f166517a96f9bcbef70a0bf19d325871f56574911"),
    "4:3": ((1024, 768), (512, 384), "c0071174302005ea5300e456d1024d47609f253f71a224dbf3d1c7b566b99dbf"),
    "3:4": ((768, 1024), (384, 512), "88400bbb17e9d80216162ad5bfed248bd9d8681954d8002ebdcdcfb4bc36a580"),
    "3:2": ((1152, 768), (576, 384), "bb076c89baed9a4e447b5b8e342a38ab870c33a19dbe72b1fc60486d8012f1ad"),
    "2:3": ((768, 1152), (384, 576), "6ea22bea1b912485a40d6337816cc0660f2b402714f52fe0c5246ede09ea3baf"),
}


def _check_fields(where: str, actual: dict, expected: dict) -> None:
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"{where}: {key} expected {value!r}, got {actual.get(key)!r}")


def _pixels(image: Image.Image) -> str:
    with image.convert("RGB") as rgb:
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def plant_fixtures(data: Path, source: Path | None = None) -> dict[str, dict]:
    """Create a new scratch gallery in a separate process with isolated settings."""
    data = data.resolve()
    source = (source or SOURCE).resolve(strict=True)
    data.mkdir()  # Only the winner can write this gallery. Never touch an existing one.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("AURADREAM_")}
    env.update(AURADREAM_DB_PATH=str(data / "auradream.db"), AURADREAM_OUTPUTS_DIR=str(data / "outputs"),
               AURADREAM_BUNDLES_DIR=str(data / "bundles"))
    subprocess.run([sys.executable, "-m", "tools.kontext_trial_fixtures", str(data), str(source)],
                   env=env, cwd=ROOT, check=True, timeout=90)
    return json.loads((data / MANIFEST).read_text(encoding="utf-8"))["fixtures"]


def _plant(data: Path, source: Path) -> None:
    from backend import storage
    from backend.schemas import GenerateRequest

    if (not data.is_dir() or any(data.iterdir()) or storage.settings.db_path != data / "auradream.db"
            or storage.settings.outputs_dir != data / "outputs"):
        raise ValueError("INP#13 planting requires the freshly reserved scratch gallery")
    fixtures = {}
    with Image.open(source) as original, original.convert("RGB") as square:
        if square.size != (1024, 1024) or _pixels(square) != CASES["1:1"][2]:
            raise ValueError("INP#13 square source does not match the frozen RGB fixture")
        storage.init_db()
        for seed, (aspect, (original_size, _, digest)) in enumerate(CASES.items()):
            with square.resize(original_size, Image.Resampling.LANCZOS) as image:
                if _pixels(image) != digest:
                    raise ValueError(f"INP#13 {aspect} original RGB does not match the frozen fixture")
                request = GenerateRequest(prompt=f"Owned synthetic house, sun, grass and red ball ({aspect})",
                                          seed=seed, width=image.width, height=image.height,
                                          steps=1, guidance_scale=1)
                row = storage.save_image(image, request, seed=seed, duration_ms=0, model_id=FIXTURE_MODEL)
            path = data / "outputs" / row.filename
            with Image.open(path) as saved:
                if saved.size != original_size or _pixels(saved) != digest or row.content_hash != digest[:16]:
                    raise ValueError(f"INP#13 {aspect} source changed during gallery planting")
            fixtures[aspect] = {"id": row.id, "filename": row.filename, "hash": digest,
                                "width": row.width, "height": row.height}
    (data / MANIFEST).write_text(json.dumps({"protocol": "inp13-kontext-fixtures-v1", "fixtures": fixtures},
                                            indent=2) + "\n", encoding="utf-8")


def _gallery_png(data: Path, name: str) -> Path:
    outputs = (data / "outputs").resolve(strict=True)
    if not isinstance(name, str) or Path(name).name != name or Path(name).suffix != ".png":
        raise ValueError("Invalid scratch gallery PNG basename")
    path = (outputs / name).resolve(strict=True)
    if path.parent != outputs:
        raise ValueError("PNG does not belong to the scratch gallery")
    return path


def verify_result(data: Path, fixture: dict, recipe: str, seed: int, job: dict, record: dict,
                  *, real: bool) -> dict[str, Any]:
    """Require the job, gallery GET, raw row and PNG to prove this exact edit."""
    source_size = (fixture["width"], fixture["height"])
    case = next((case for case in CASES.values() if case[0] == source_size), None)
    if case is None or fixture["hash"] != case[2]:
        raise ValueError("Unrecognized INP#13 source proof")
    if recipe == "kontext-v1":
        width, height = case[1]
    elif recipe == "kontext-1024-v1" and source_size == (1024, 1024):
        width, height = 1024, 1024
    else:
        raise ValueError("Unsupported INP#13 recipe/source combination")
    proof = {key: fixture[key] for key in ("filename", "hash", "width", "height")}
    edit = {"source": fixture["id"], "recipe": recipe, "edit_source": proof}
    _check_fields("job", job, {"status": "done", "prompt": INSTRUCTION, "edit": edit,
                               "width": width, "height": height, "image_count": len(job["images"])})
    matching = [item for item in job["images"] if item["id"] == record["id"]]
    if len(matching) != 1:
        raise ValueError("Job must contain exactly one matching gallery result")
    uri = (data / "auradream.db").resolve(strict=True).as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        source = connection.execute("SELECT * FROM images WHERE id=?", (fixture["id"],)).fetchone()
        output = connection.execute("SELECT * FROM images WHERE id=?", (record["id"],)).fetchone()
    if source is None or output is None:
        raise ValueError("Source or result is missing from scratch gallery")
    _check_fields("source SQL", dict(source), {"filename": proof["filename"], "content_hash": proof["hash"][:16],
                                              "width": proof["width"], "height": proof["height"], "binned_at": None})
    with Image.open(_gallery_png(data, proof["filename"])) as original:
        if original.size != source_size or _pixels(original) != proof["hash"]:
            raise ValueError("Original source no longer matches its full RGB proof")
    sql = dict(output)
    sql["edit"] = json.loads(sql["edit"])
    path = _gallery_png(data, sql["filename"])
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError("Encoded PNG dimensions do not match the trial case")
        rgb = _pixels(image)
        metadata = json.loads(image.info["auradream"])

    shared = {"prompt": INSTRUCTION, "seed": seed, "steps": 28, "guidance_scale": 2.5, "batch": 1,
              "quantization": "fp8", "edit": edit, "negative_prompt": None, "mu": None, "true_cfg": None}
    peaks = {key: sql[key] for key in ("vram_peak_allocated", "vram_peak_reserved")}
    allocated, reserved = peaks.values()
    if real or allocated is not None or reserved is not None:
        if type(allocated) is not int or type(reserved) is not int or not 0 < allocated <= reserved:
            raise ValueError("Missing or inconsistent measured allocator peaks")
    if not job["run_id"] or not job["job_id"]:
        raise ValueError("Missing run or job identity")
    fields = {**shared, **peaks, "id": record["id"], "filename": path.name, "width": width, "height": height,
              "content_hash": rgb[:16], "model_id": MODEL, "job_id": job["job_id"], "run_id": job["run_id"],
              "duration_ms": sql["duration_ms"]}
    for label, evidence in (("SQL", sql), ("gallery API", record), ("job image", matching[0])):
        _check_fields(label, evidence, fields)
    _check_fields("PNG metadata", metadata, {**shared, "model": MODEL, "job": job["job_id"], "run": job["run_id"],
                                             "size": f"{width}x{height}", "hash": rgb[:16]})
    if any(metadata.get(key) != value for key, value in peaks.items()):
        raise ValueError("PNG allocator evidence disagrees with the gallery")
    if real and recipe == "kontext-v1" and source_size == (1024, 1024) and seed == 12345 and rgb != CONTROL_RGB:
        raise ValueError(f"512-square control expected {CONTROL_RGB}, got {rgb}")
    return {"image_id": sql["id"], "path": str(path), "rgb_sha256": rgb, "edit": edit,
            "dimensions": [width, height], "seed": seed, "instruction": INSTRUCTION, "job_id": job["job_id"],
            "run_id": job["run_id"], "duration_ms": sql["duration_ms"], **peaks}


if __name__ == "__main__":
    _plant(Path(sys.argv[1]), Path(sys.argv[2]))
