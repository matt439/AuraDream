"""Repeat the finite public-image CPU detector trial, never the private gallery.

uv run python tools/detailer_trial.py --download --work work/detailer-trial
Without --download, every source and detector asset must already be present.
The frozen manifest includes the licences, exact bytes and approximate labels;
its original report remains evidence, rather than an automatically updated test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
import psutil  # noqa: E402

from backend.detailer_detectors import Asset, DETECTORS, DetailerDetectors, _verified  # noqa: E402

MANIFEST = ROOT / "docs/maintainer/detailer-cpu-2026-11-16.json"


def _iou(a: list[int], b: list[int]) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / total if total else 0.0


def _matches(expected: list[list[int]], regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy maximum IoU, deterministic label/result index tie-break, >=0.3."""
    pairs = sorted(((-_iou(box, region["box"]), i, j)
                    for i, box in enumerate(expected) for j, region in enumerate(regions)))
    labels: set[int] = set()
    found: set[int] = set()
    matches = []
    for negative_iou, i, j in pairs:
        if -negative_iou >= 0.3 and i not in labels and j not in found:
            labels.add(i)
            found.add(j)
            matches.append({"label": i, "region": j, "iou": -negative_iou})
    return {"matches": matches, "misses": len(expected) - len(labels), "false_positives": len(regions) - len(found)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly fetch the pinned public images and detector assets")
    parser.add_argument("--work", type=Path, default=ROOT / "work/detailer-trial")
    parser.add_argument("--report", type=Path, help="Output JSON (default: <work>/report.json)")
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    images = args.work / "images"
    images.mkdir(parents=True, exist_ok=True)
    adapter = DetailerDetectors(args.work / "assets")
    try:
        for source in manifest["sources"]:
            name = source["file"]
            if Path(name).name != name or not 0 < source["bytes"] <= 3_000_000:
                raise ValueError("The frozen trial contains an invalid source name/size.")
            asset = Asset(name, source["url"], source["bytes"], source["sha256"], "public-trial-v1")
            path = images / name
            if not _verified(path, asset, lambda: False):
                if not args.download:
                    raise ValueError(f"Missing or changed trial image {name}; repeat with --download.")
                adapter._download(path, asset, lambda: False)
        if args.download:
            for detector in DETECTORS:
                adapter.prepare(detector.id)
        process = psutil.Process()
        samples: list[int] = []
        stopping = threading.Event()

        def sample() -> None:
            while not stopping.is_set():
                samples.append(process.memory_info().rss)
                time.sleep(0.01)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        report: dict[str, Any] = {
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "platform": platform.platform(), "python": platform.python_version(),
            "source_code_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("backend/detailer_detectors.py", "backend/detailer_hands.py", "tools/detailer_trial.py")},
            "method": manifest["method"], "runs": [],
        }
        try:
            for name, label in manifest["labels"]["images"].items():
                if name == "blank":
                    image = Image.new("RGB", tuple(label["size"]))
                else:
                    with Image.open(images / name.split(":")[0]) as original:
                        original.seek(label.get("frame", 0))
                        image = original.convert("RGB")
                    if "crop" in label:
                        image = image.crop(tuple(label["crop"]))
                for detector in DETECTORS:
                    started, sample_start = time.perf_counter(), len(samples)
                    run: dict[str, Any] = {"image": name, "detector": detector.id, "source_size": list(image.size)}
                    try:
                        regions, facts = adapter.detect(detector.id, image)
                        run["regions"] = [region.model_dump(mode="json") for region in regions]
                        run["facts"] = [fact.model_dump(mode="json") for fact in facts]
                        label_key = "faces" if detector.kind == "face" else "hands"
                        if label_key in label:
                            run.update(_matches(label[label_key], run["regions"]))
                        else:
                            run["evaluation"] = "No predeclared labels for this capability/source; inference evidence only."
                    except Exception as exc:
                        run["error"] = str(exc)
                    run["elapsed_seconds"] = time.perf_counter() - started
                    run["rss_peak_sampled"] = max(samples[sample_start:] or [process.memory_info().rss])
                    report["runs"].append(run)
                    print(f"{name}: {detector.id}: {run.get('error', str(len(run.get('regions', []))) + ' regions')}", flush=True)
        finally:
            stopping.set()
            sampler.join(timeout=1)
        report["process_peak_sampled"] = max(samples or [process.memory_info().rss])
        target = args.report or args.work / "report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {target}")
        return int(any("error" in run for run in report["runs"]))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Detailer CPU trial refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
