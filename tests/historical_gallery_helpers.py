"""Independent expectations from the synthetic inputs to two historical writers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from backend import storage

ARCHIVES = Path(__file__).parent / "fixtures" / "historical_gallery"
KREA = "black-forest-labs/FLUX.1-Krea-dev"
KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
STAMP = "2026-01-02T03:04:06+00:00"

# Intentionally independent of the writer utility and current serializers.
SAMPLES = {
    101: ("red study", (110, 20, 30), "2e0313b40a4c"),
    102: ("blue refinement", (30, 40, 130), "0c5c3f189b3e"),
    103: ("green discard", (11, 95, 23), "64c5959b18fa"),
    201: ("teal square", (17, 80, 93), "208f9c4ec76e"),
    202: ("paint it ochre", (118, 62, 31), "479ebfb400df"),
}
RUN_IDS = {seed: sample[2] for seed, sample in SAMPLES.items()}


def filename(seed):
    return f"20260102-030406-{seed}-{seed:06d}.png"


def _rgb_digest(seed):
    return hashlib.sha256(bytes(SAMPLES[seed][1]) * (512 * 512)).hexdigest()


def expected_edit(source_id):
    return {
        "source": source_id, "recipe": "kontext-v1",
        "edit_source": {
            "filename": filename(201), "hash": _rgb_digest(201), "width": 512, "height": 512,
        },
    }


def assert_gallery(version, source_id, *, png_only=False, orphan=False, expected_chunk="auradream"):
    """Check the promised SQL and PNG fields, including the SQL-only losses of PNG recovery."""
    seeds = {"legacy-v1": (101, 102, 103), "edit-v2": (201, 202)}[version]
    if orphan:
        assert version == "edit-v2" and source_id is None
        seeds = (202,)
    found = {}
    for seed in seeds:
        prompt, _, run_id = SAMPLES[seed]
        binned = seed == 103 and not png_only
        matches, total, _ = storage.list_images(search=f"seed:{seed}", binned=binned)
        assert total == 1
        record, = matches
        found[seed] = record
        assert record.filename == filename(seed)
        assert (record.prompt, record.steps, record.guidance_scale) == (prompt, 28, 2.5 if seed == 202 else 3.5)
        model = KONTEXT if seed == 202 else KREA
        assert (record.model_id, record.quantization, record.batch) == (model, "fp8", 1)
        assert (record.width, record.height) == (512, 512)
        assert record.created_at == STAMP
        assert record.duration_ms == (0 if png_only else seed + 10)
        assert record.favorite == (seed in (101, 201) and not png_only)
        assert record.binned_at == (STAMP if binned else None)
        assert (record.job_id, record.run_id) == (f"fixture-job-{seed}", run_id)
        negative = "glare" if seed == 101 else None
        true_cfg = 1.75 if seed == 101 else None
        variables = {"color": "red"} if seed == 101 else None
        assert (record.negative_prompt, record.true_cfg, record.variables) == (negative, true_cfg, variables)

        proof = _rgb_digest(seed)
        assert record.content_hash == proof[:16]
        with Image.open(storage.settings.outputs_dir / record.filename) as png:
            assert (png.mode, png.size) == ("RGB", (512, 512))
            assert hashlib.sha256(png.tobytes()).hexdigest() == proof
            assert {"auradream", "local-diffusion", "krea"}.intersection(png.info) == {expected_chunk}
            meta = json.loads(png.info[expected_chunk])
        assert (meta["prompt"], meta["seed"], meta["model"], meta["hash"]) == (prompt, seed, model, proof[:16])
        assert (meta["steps"], meta["guidance_scale"], meta["quantization"], meta["batch"]) == (
            28, record.guidance_scale, "fp8", 1,
        )
        assert (meta["job"], meta["run"]) == (record.job_id, record.run_id)
        assert (meta["negative_prompt"], meta["true_cfg"]) == (negative, true_cfg)
        assert meta["variables"] == variables
        assert record.refine == meta.get("refine") == (source_id if seed == 102 else None)
        assert record.strength == meta.get("strength") == (0.375 if seed == 102 else None)
        if seed == 102:
            assert meta["refine_source"] == {
                "filename": filename(101), "hash": _rgb_digest(101)[:16], "width": 512, "height": 512,
            }
        edit = record.edit.model_dump(mode="json") if record.edit else None
        assert edit == meta.get("edit") == (expected_edit(source_id) if seed == 202 else None)
        assert record.edit_missing == (seed == 202 and orphan)
    if not orphan:
        assert found[seeds[0]].id == source_id
    return found
