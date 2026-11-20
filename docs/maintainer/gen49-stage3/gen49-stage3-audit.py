"""Independent CPU replay of captured GEN#49 stage-three card tiles and native inspection views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

from PIL import Image, ImageChops, ImageDraw, ImageOps

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
WORK = ROOT / "work"
HISTORICAL = ROOT / "docs/maintainer/gen49-sdxl-four-region-2026-11-17.json"
SCENES = ("architecture", "organic", "lettering", "subject")
RUNS = (*SCENES, "repeat", "batch", "solo-next", "cancel", "recovery", "historic")
NATIVE = {"vertical": (896, 0, 1152, 2048), "horizontal": (0, 896, 2048, 1152), "center": (768, 768, 1280, 1280)}
TIMES = [281, 246, 211, 176, 141, 106, 71, 36, 1]


def parsed(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(bytes_):
    return hashlib.sha256(bytes_).hexdigest()


def pixels(path, mode, size):
    with Image.open(path) as handle:
        assert handle.mode == mode and handle.size == size, (path, handle.mode, handle.size)
        return handle.copy()


def equal(a, b):
    return a.mode == b.mode and a.size == b.size and ImageChops.difference(a, b).getbbox() is None


def facts(image):
    return digest(image.tobytes())


def mask_for(tile):
    # Independent integer ramp encoded as a row, then mirrored and min-combined.
    x1, y1, x2, y2 = tile["paint_box"]
    assert x2 - x1 == y2 - y1 == 1088
    white_to_black = bytes(min(255, max(1, (1088 - i) * 255 // 64)) for i in range(1088))
    x = Image.frombytes("L", (1088, 1), white_to_black if x1 == 0 else white_to_black[::-1])
    y = Image.frombytes("L", (1, 1088), white_to_black if y1 == 0 else white_to_black[::-1])
    mask = Image.new("L", (2048, 2048))
    mask.paste(ImageChops.darker(x.resize((1088, 1088)), y.resize((1088, 1088))), (x1, y1))
    assert facts(mask) == tile["mask_rgb_bytes_sha256"]
    assert digest(b"L 2048x2048\n" + mask.tobytes()) == tile["mask_hash"]
    return mask


def pipeline_mask(mask, box):
    # These rectangular selections stay rectangular after threshold, crop, BOX,
    # square dilation of eight working pixels, and outward 8x8 latent grid snapping.
    levels = [0] + [255] * 255
    initial = mask.point(levels).crop(box).resize((1040, 1040), Image.Resampling.BOX)
    left, top, right, bottom = initial.getbbox()

    def lo(v):
        return max(0, (v - 8) // 8 * 8)

    def hi(v):
        return min(1040, ((v + 8 + 7) // 8) * 8)

    result = Image.new("L", (1040, 1040))
    result.paste(255, (lo(left), lo(top), hi(right), hi(bottom)))
    return result


def overlay(source, raw, mask, box):
    # Diffusers 0.42.1 applies the premultiplied source's inverted mask and
    # converts back to straight alpha before alpha-compositing. A single
    # Image.composite would round some feathered pixels differently.
    pasted = source.copy()
    pasted.paste(raw.resize((1120, 1120), Image.Resampling.LANCZOS), box)
    kept = Image.new("RGBa", source.size)
    kept.paste(source.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask))
    return Image.alpha_composite(pasted.convert("RGBA"), kept.convert("RGBA")).convert("RGB")


def audit_run(folder, name, fixture, history, masks, sigma_ref):
    call = parsed(folder / "call.json")
    proof = call["proof"]
    assert proof["root_source"]["hash"] == fixture["rgb_sha256"]
    assert proof["root_source"]["width"] == proof["root_source"]["height"] == 1024
    assert proof["preparation"] == "lanczos-2x-rgb-v1"
    original = pixels(Path(fixture["source_path"]), "RGB", (1024, 1024))
    assert facts(original) == fixture["rgb_sha256"]
    root = pixels(folder / "prepared.png", "RGB", (2048, 2048))
    assert equal(root, original.resize(root.size, Image.Resampling.LANCZOS))
    assert facts(root) == proof["prepared_rgb_sha256"] == fixture["prepared_rgb_sha256"]
    expected_seed = fixture["seed"] + int(name == "solo-next")
    assert call["seed"] == expected_seed and call["count"] == (2 if name == "batch" else 1)
    cancelled = name == "cancel"
    assert len(call["tiles"]) == (2 if cancelled else call["count"] * 4)
    assert call["progress"] == [[i, call["count"] * 36] for i in range(1, 13 if cancelled else call["count"] * 36 + 1)]
    assert call["root_checks"] == [fixture["rgb_sha256"]] * (2 if cancelled else call["count"] * 5)
    assert len(call["results"]) == (0 if cancelled else call["count"])
    assert call["prompt"] == fixture["prompt"]
    assert proof["root_source"]["filename"] == Path(proof["root_source"]["filename"]).name
    original_gallery = pixels(folder.parent / "outputs" / proof["root_source"]["filename"], "RGB", (1024, 1024))
    assert equal(original, original_gallery)
    if cancelled:
        assert call["job"]["status"] == "cancelled" and not call["job"]["images"]
        assert call["unchanged_gallery"] and not call["job"]["failed"] and not call["job"]["completed"]
    tiles, final, historical = [], [], []
    current = root
    for i, row in enumerate(call["tiles"]):
        ordinal = i % 4
        if ordinal == 0:
            current = root
        historical_tile = history["plan"]["tiles"][ordinal]
        if not cancelled:
            plan = row["plan"]
            for key in ("name", "ordinal", "paint_box", "crop_box", "working", "mask_hash"):
                assert plan[key] == historical_tile[key], (name, i, key)
            assert plan["mask_bytes_sha256"] == historical_tile["mask_rgb_bytes_sha256"]
            assert plan["seed"] == row["seed"]
        assert row["seed"] == expected_seed + i // 4 + ordinal
        assert row["generator_device"] == "cpu" and row["prompt"] == fixture["prompt"]
        assert row["negative"] == "low quality, blurry" and row["strength"] == 0.35
        assert row["cfg"] == 5 and row["full_steps"] == 28
        assert row["scheduler"] == "EulerDiscreteScheduler" and row["begin_index"] == 19
        steps = row["steps"]

        def path(item, tile_index=i + 1):
            return folder / f"tile-{tile_index:02d}-{item}.png"

        partial = cancelled and i == 1
        assert len(steps) == (3 if partial else 9)
        for j, step in enumerate(steps):
            ix, tm, sig = step["index"], step["timestep"], step["sigma"]
            assert ix == 19 + j and tm == TIMES[j] and math.isfinite(sig) and sig > 0, (ix, tm, sig)
            assert step["step"] == j + 1 and tm == row["timesteps"][ix]
            assert sig == row["schedule"][ix] and step["next_sigma"] == row["schedule"][ix + 1]
            if j:
                assert sig < steps[j - 1]["sigma"]
            assert sigma_ref.setdefault(ix, sig) == sig
        box = tuple(historical_tile["crop_box"])
        mask, model_mask = masks[ordinal]
        oracle_input = current.crop(box).resize((1040, 1040), Image.Resampling.LANCZOS)
        observed_input = pixels(path("input"), "RGB", (1040, 1040))
        observed_mask = pixels(path("mask"), "L", (1040, 1040))
        assert equal(oracle_input, observed_input) and equal(model_mask, observed_mask)
        tile_facts = {
            "ordinal": ordinal,
            "seed": row["seed"],
            "callbacks": len(steps),
            "source_rgb": facts(current),
            "input_rgb": facts(observed_input),
            "pipeline_mask": facts(observed_mask),
        }
        assert facts(observed_input) == row["input_sha256"] and facts(observed_mask) == row["mask_bytes_sha256"]
        tiles.append(tile_facts)
        if partial:
            assert not path("raw").exists() and not path("composite").exists()
            continue
        raw = pixels(path("raw"), "RGB", (1040, 1040))
        intermediate = pixels(path("composite"), "RGB", (2048, 2048))
        oracle = overlay(current, raw, mask, box)
        assert equal(oracle, intermediate), (name, i, "raw replay")
        raw_diff = ImageChops.difference(current, intermediate)
        outside = mask.point([255] + [0] * 255)
        # Test every zero-alpha pixel of the direct source.
        assert max(ImageChops.multiply(channel, outside).getextrema()[1] for channel in raw_diff.split()) == 0
        tile_facts.update(raw_rgb=facts(raw), composite_rgb=facts(intermediate), outside_mask_max=0)
        assert row["raw_sha256"] == facts(raw) and row["composite_sha256"] == facts(intermediate)
        assert row["source_sha256"] == facts(current)
        if not cancelled:
            result = call["results"][i // 4]
            tr = result["tiles"][ordinal]
            assert result["source_rgb_sha256"] == facts(root) and tr["plan"] == row["plan"]
            assert tr["source_rgb_sha256"] == facts(current) and tr["result_rgb_sha256"] == facts(intermediate)
            assert tr["callback_count"] == 9 and tr["raw_size"] == [1040, 1040]
        if name == "historic":
            old = next(c for c in history["cases"] if c.get("id") == history["serial_chain"][ordinal]["result"])
            historical.append(
                {
                    "raw_matches": old["raw_working_output"]["pixel_sha256"] == facts(raw),
                    "composite_matches": old["pixel_sha256"] == facts(intermediate),
                    "expected_raw_rgb": old["raw_working_output"]["pixel_sha256"],
                    "expected_composite_rgb": old["pixel_sha256"],
                }
            )
        current = intermediate
        if ordinal == 3:
            image = pixels(folder / f"final-{1 + i // 4}.png", "RGB", (2048, 2048))
            assert equal(image, current)
            final.append(facts(image))
            if name == "recovery":
                job = call["job"]
                assert job["status"] == "done" and job["seed"] == call["seed"]
                with Image.open(folder.parent / "outputs" / job["filename"]) as saved:
                    chunk = json.loads(saved.info["auradream"])
                    assert equal(saved, image) and chunk["tiled_refine"] == proof
    assert len(list(folder.glob("final-*.png"))) == len(final)
    return {
        "tiles": tiles,
        "final_rgb": final,
        "root_rgb": facts(original),
        "prepared_rgb": facts(root),
        "historical_control": historical,
    }


def views(root, target, frozen_boxes):
    overview = Image.new("RGB", (4 * 300, 2 * 332), "white")
    labels = ImageDraw.Draw(overview)
    manifest = {}
    for i, name in enumerate(SCENES):
        out = target / name
        out.mkdir()
        im = [
            pixels(root / name / "prepared.png", "RGB", (2048, 2048)),
            pixels(root / name / "final-1.png", "RGB", (2048, 2048)),
        ]
        manifest[name] = {}
        for j, description in enumerate(("source", "final")):
            overview.paste(im[j].resize((300, 300), Image.Resampling.LANCZOS), (300 * i, 332 * j + 25))
            labels.text((300 * i + 6, 332 * j + 4), name + " / " + description, fill="black")
        for kind, box in (NATIVE | {f"fixture-{k}": v for k, v in frozen_boxes.items()}).items():
            crops = [s.crop(box) for s in im]
            for tag, img in zip(("source", "final"), crops):
                file = out / f"{tag}-{kind}.png"
                img.save(file)
                manifest[name][file.name] = digest(file.read_bytes())
            w, h = crops[0].size
            stacked = "horizontal" in kind
            pair = Image.new("RGB", (w, 2 * h + 8) if stacked else (2 * w + 8, h), "#404040")
            pair.paste(crops[0], (0, 0))
            pair.paste(crops[1], (0, h + 8) if stacked else (w + 8, 0))
            pair.save(out / f"pair-{kind}.png")
    overview.save(target / "overview.png")
    return manifest


def main():
    if not __debug__:
        raise RuntimeError("GEN#49 pixel verification requires Python assertions enabled.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.attempt.resolve(strict=True)
    out = args.output or root / "audit"
    out.mkdir(parents=False, exist_ok=False)
    history = parsed(HISTORICAL)
    fixtures_dir = WORK / "gen49-stage3-fixtures"
    frozen = parsed(fixtures_dir / "manifest.json")
    fixtures = {r["name"]: dict(r, source_path=str(fixtures_dir / r["filename"])) for r in frozen["scenes"]}
    old_raw = Path(os.environ.get("GEN49_HISTORY", WORK / Path(history["raw_directory"]).name))
    fixtures["historic"] = {
        "seed": 12346,
        "prompt": history["plan"]["recipe"]["prompt"],
        "rgb_sha256": history["cases"][0]["pixel_sha256"],
        "source_path": str(old_raw / "baseline.png"),
        "prepared_rgb_sha256": history["cases"][2]["source_full_rgb_sha256"],
    }
    masks = []
    for tile in history["plan"]["tiles"]:
        mask = mask_for(tile)
        masks.append((mask, pipeline_mask(mask, tuple(tile["crop_box"]))))
    rows, sigmas = {}, {}
    for name in RUNS:
        print(f"Replaying {name}", flush=True)
        rows[name] = audit_run(root / name, name, fixtures.get(name, fixtures["architecture"]), history, masks, sigmas)
    base = rows["architecture"]
    matches = {
        "repeat": base["tiles"] == rows["repeat"]["tiles"],
        "recovery": base["tiles"] == rows["recovery"]["tiles"],
        "completed_cancelled_prefix": rows["cancel"]["tiles"][0] == base["tiles"][0],
        "batch_first": base["tiles"] == rows["batch"]["tiles"][:4],
        "batch_second": rows["solo-next"]["tiles"] == rows["batch"]["tiles"][4:],
        "batch_seeds_distinct": len(set(rows["batch"]["final_rgb"])) == 2,
    }
    report = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "matches": matches,
        "runs": rows,
        "sigma_by_schedule_index": sigmas,
        "native_boxes": NATIVE,
        "views": views(root, out, frozen["frozen_native_boxes_2048"]),
        "framework_free": "torch" not in sys.modules,
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert all(matches.values()) and report["framework_free"], matches


if __name__ == "__main__":
    main()
