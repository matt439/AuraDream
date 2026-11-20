"""Read the five original image suites in the 2026-11-18 qualification receipts.

The tracked receipts authenticate each raw measurements file. Its completed row
binds a fixed case, execution arm, elapsed time and RGB digest. The pack builder
checks the digest on the exact source pixels it copies.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import perf_cases

_RECEIPTS = Path(__file__).resolve().parents[1] / "docs" / "maintainer"


@dataclass(frozen=True)
class Side:
    method: str
    seconds: float
    path: Path
    rgb_sha256: str


@dataclass(frozen=True)
class Pair:
    family: str
    model: str
    seed: int
    prompt: str
    width: int
    height: int
    left: Side
    right: Side


@dataclass(frozen=True)
class _Suite:
    receipt: str
    directory: str

    def images(self, root: Path) -> dict[str, dict[str, Any]]:
        receipt = json.loads((_RECEIPTS / self.receipt).read_text(encoding="utf-8"))
        regional = self.directory.startswith("perf1/")
        loader = receipt["loader"] if regional else receipt["runtime"]["loader"]
        for field, expected in (("checkpoint", perf_cases.CHECKPOINT), ("quantization", perf_cases.QUANTIZATION),
                                ("offload", perf_cases.OFFLOAD)):
            if loader[field] != expected:
                raise ValueError(f"Measured {field} differs from the fixed recipe in {self.receipt}")
        prefix = f".artifacts/gpu-20261118/{self.directory}"
        if regional:
            name = "measurements.json"
            digest, = (
                item["sha256"] for session in receipt["sessions"] for item in session["evidence"]
                if item["path"] == f"{prefix}/{name}"
            )
        else:
            name = "measurements.jsonl"
            digest, = (
                session["raw_artifact_sha256"][name] for session in receipt["sessions"]
                if session["session"] == prefix
            )
        path = root / self.directory / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Measurement SHA256 differs from {self.receipt}: {path}")
        records = json.loads(data) if name.endswith(".json") else [json.loads(line) for line in data.splitlines()]
        images = {}
        for row in records:
            if row["kind"] == "image":
                case_id = row["case"]["id"]
                if case_id in images:
                    raise ValueError(f"Duplicate measured image {case_id} in {path}")
                images[case_id] = row
        return images


_REGIONAL = "perf1-completed-matrix-2026-11-18.json"
_ATTENTION = "perf3-qualification-2026-11-18.json"
_CACHE = "perf4-qualification-2026-11-18.json"


@dataclass(frozen=True)
class _Arm:
    method: str
    suite: _Suite
    execution_field: str | None = None
    recipe_name: str | None = None
    cache_suffix: str = ""

    def execution(self) -> dict[str, str | None] | None:
        if self.execution_field is None:
            return None
        fields: dict[str, str | None] = dict.fromkeys(perf_cases.RECIPE_FLAGS)
        fields[self.execution_field] = self.recipe_name
        return fields


_FBC_SUITE = _Suite(_CACHE, "perf4-fbc-generate-suite-a1")
_FAMILIES = (
    ("Regional compilation", ("A", "B"),
     _Arm("Eager", _Suite(_REGIONAL, "perf1/eager-suite-a1")),
     _Arm("Regional", _Suite(_REGIONAL, "perf1/regional-suite-retained-a1"), "compilation", "regional-v1")),
    ("Attention backend", ("A", "B"),
     _Arm("Native", _Suite(_ATTENTION, "perf3-native-generate-suite-a1")),
     _Arm("cuDNN", _Suite(_ATTENTION, "perf3-cudnn-generate-suite-a1"), "attention_backend", "cudnn-v1")),
    ("Step caching", ("A",),
     _Arm("Off", _FBC_SUITE, cache_suffix="-off"),
     _Arm("On", _FBC_SUITE, "step_cache", "fbc-005-v1", "-on")),
)


def _image(
    root: Path, arm: _Arm, images: dict[str, dict[str, Any]], shape: str, prompt: str, seed: int,
) -> tuple[Side, dict[str, Any]]:
    # This first occurrence is the actual timed pair, even where all the later
    # same-seed repeats have identical RGB bytes.
    first = (shape, prompt, seed) == ("A", "scene", 4471)
    case_id = ("A-first" if first else f"{shape}-{prompt}-{seed}") + arm.cache_suffix
    path = root / arm.suite.directory / f"{case_id}.png"
    if case_id not in images:
        raise ValueError(f"Missing measured fixed case for {path}")
    row = images[case_id]
    expected_case = {"id": case_id, "shape": shape, "prompt": prompt, "seed": seed}
    if arm.cache_suffix:
        expected_case["cached"] = arm.cache_suffix == "-on"
    width, height = perf_cases.SHAPES[shape]
    applied = dict(row["request_applied"])
    execution = arm.execution()
    # Only FBC is request-owned; Regional and cuDNN were privately prepared loads.
    applied_execution = applied.pop("execution")
    if (row["case"] != expected_case or row.get("seed", seed) != seed or row["status"] != "completed"
            or row["execution"] != execution
            or applied_execution != (execution if arm.cache_suffix else None)):
        raise ValueError(f"Measured case, status or execution arm differs for {path}")
    expected_request = {
        "prompt": perf_cases.PROMPTS[prompt], "width": width, "height": height, "steps": perf_cases.STEPS,
        "guidance_scale": perf_cases.CFG, "num_images": perf_cases.BATCH,
    }
    if any(applied[key] != value for key, value in expected_request.items()):
        raise ValueError(f"Measured request differs from fixed case for {path}")
    seconds, digest = row["seconds"], row["pixel_sha256"]
    if not math.isfinite(seconds) or seconds <= 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Missing or invalid measured time or RGB digest for {path}")
    return Side(arm.method, seconds, path, digest), applied


def load_pairs(root: Path) -> list[Pair]:
    """Return the 12 regional, 12 attention and six cache pairs from original suite runs."""
    pairs = []
    for family, shapes, left_arm, right_arm in _FAMILIES:
        left_images = left_arm.suite.images(root)
        right_images = left_images if right_arm.suite == left_arm.suite else right_arm.suite.images(root)
        for shape in shapes:
            width, height = perf_cases.SHAPES[shape]
            for prompt in perf_cases.PROMPTS:
                for seed in perf_cases.SEEDS:
                    left, request = _image(root, left_arm, left_images, shape, prompt, seed)
                    right, other_request = _image(root, right_arm, right_images, shape, prompt, seed)
                    if request != other_request:
                        raise ValueError(f"Unmatched applied requests in {family}, {shape}/{prompt}/{seed}")
                    pairs.append(Pair(family, perf_cases.CHECKPOINT, seed, request["prompt"], width, height, left, right))
    if len(pairs) != 30:
        raise ValueError(f"The fixed 2026-11-18 suite has 30 pairs, found {len(pairs)}")
    return pairs
